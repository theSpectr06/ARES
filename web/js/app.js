"use strict";

/* ================================================================
   ARES — Mission Planner  ·  app.js
   ================================================================ */

const DATA_DIRECTORY   = "./data/";
const DEFAULT_COST_OPACITY = 0.72;
const MAP_PADDING      = 24;

// ----------------------------------------------------------------
// Application state
// ----------------------------------------------------------------
const state = {
    map: null,
    mission: null,
    route: null,
    destinations: [],
    scienceSites: null,
    selectedDestination: null,
    activeView: "regional",
    metersPerPxOverview: null,  // metres per overview image pixel
    metersPerPxDetail: null,    // metres per detail image pixel (= raster pixel)
    gridLayer: null,            // persistent metric grid layer
    layers: {
        regionalImagery: null,
        detailImagery: null,
        cost: null,
        routeCasing: null,
        routeLine: null,
        endpoints: null,
        destinationPins: null,
        basecampPin: null,
    },
};

const byId = (id) => document.getElementById(id);

// ----------------------------------------------------------------
// JSON / asset utilities
// ----------------------------------------------------------------
async function loadJSON(filename) {
    const response = await fetch(`${DATA_DIRECTORY}${filename}`);
    if (!response.ok) {
        throw new Error(`Failed to load ${filename}: HTTP ${response.status}`);
    }
    return response.json();
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

function safeExternalURL(value) {
    try {
        const url = new URL(value);
        if (url.protocol === "https:" || url.protocol === "http:") return url.href;
    } catch { /* invalid */ }
    return null;
}

function imageURL(filename) {
    return new URL(filename, new URL(DATA_DIRECTORY, window.location.href)).href;
}

async function verifyImage(url, expectedWidth, expectedHeight) {
    const image = new Image();
    image.src = url;
    try { await image.decode(); }
    catch { throw new Error(`Could not decode map asset: ${url}`); }
    if (image.naturalWidth !== expectedWidth || image.naturalHeight !== expectedHeight) {
        throw new Error(
            `Image dimensions mismatch. Expected ${expectedWidth}×${expectedHeight}; ` +
            `received ${image.naturalWidth}×${image.naturalHeight}.`
        );
    }
}

// ----------------------------------------------------------------
// Unit formatting
// ----------------------------------------------------------------
/**
 * Format a metre value into a human-readable metric string.
 * Uses km for distances ≥ 1 km, metres otherwise.
 */
function formatDistance(meters) {
    if (!Number.isFinite(meters)) return "—";
    if (meters >= 10000) return `${(meters / 1000).toFixed(1)} km`;
    if (meters >= 1000)  return `${(meters / 1000).toFixed(2)} km`;
    if (meters >= 100)   return `${Math.round(meters)} m`;
    return `${meters.toFixed(1)} m`;
}

// ----------------------------------------------------------------
// Coordinate transforms
// ----------------------------------------------------------------
function imageBounds(width, height) {
    return [[0, 0], [height, width]];
}

function routeToLatLng(coordinates, imageHeight) {
    return coordinates.map((point) => [imageHeight - point.y, point.x]);
}

function mapPointToImagePixel(latlng, imageHeight) {
    return { x: latlng.lng, y: imageHeight - latlng.lat };
}

function getOverviewTransform() {
    const { mission } = state;
    const overview = mission.overview;
    const raster = mission.raster;

    if (Array.isArray(overview.transform) && overview.transform.length >= 6) {
        return overview.transform;
    }

    const scaleX = raster.width / overview.width;
    const scaleY = raster.height / overview.height;
    const [a, b, c, d, e, f] = raster.transform;

    return [a * scaleX, b * scaleY, c, d * scaleX, e * scaleY, f];
}

function pixelToMars(x, y, transform) {
    const [a, b, c, d, e, f] = transform;
    return { x: a * x + b * y + c, y: d * x + e * y + f };
}

/**
 * Returns the basecamp position as a Leaflet [lat, lng] pair
 * in the regional (overview) image coordinate system.
 * Derived from the route's starting point plus the detail-crop offsets.
 */
function getBasecampLatLng() {
    if (!state.mission || !state.route?.coordinates?.length) return null;

    const start  = state.route.coordinates[0];
    const detail = state.mission.detail;
    const overview = state.mission.overview;
    const raster   = state.mission.raster;

    // Detail image pixel → raster pixel
    const rasterX = detail.column_offset + start.x;
    const rasterY = detail.row_offset    + start.y;

    // Raster pixel → overview image pixel
    const overviewX = rasterX * (overview.width  / raster.width);
    const overviewY = rasterY * (overview.height / raster.height);

    // Overview pixel → Leaflet CRS.Simple [lat, lng]
    return [overview.height - overviewY, overviewX];
}

function updateCoordinateReadout(latlng) {
    const output = byId("map-coordinate-readout");
    if (!output || !state.mission) return;

    const { mission, activeView } = state;

    const pixel = activeView === "regional"
        ? mapPointToImagePixel(latlng, mission.overview.height)
        : mapPointToImagePixel(latlng, mission.detail.height);

    let rasterPixel;
    let transform;

    if (activeView === "regional") {
        transform    = getOverviewTransform();
        rasterPixel  = pixel;
    } else {
        const detail = mission.detail;
        rasterPixel  = { x: pixel.x + detail.column_offset, y: pixel.y + detail.row_offset };
        transform    = mission.raster.transform;
    }

    const mars = pixelToMars(rasterPixel.x, rasterPixel.y, transform);
    output.textContent = `E\u202f${mars.x.toFixed(1)}\u202fm · N\u202f${mars.y.toFixed(1)}\u202fm`;
}

// ----------------------------------------------------------------
// Metric grid layer
// ----------------------------------------------------------------
const MetricGridLayer = L.Layer.extend({

    initialize(options) {
        L.setOptions(this, options);
        this._mpp = (options && options.metersPerPixel) || 1;
    },

    onAdd(map) {
        const wrapper = document.createElement("div");
        wrapper.className = "metric-grid-layer";
        Object.assign(wrapper.style, {
            position: "absolute",
            inset: "0",
            pointerEvents: "none",
            zIndex: "450",
            overflow: "hidden",
        });

        const canvas = document.createElement("canvas");
        Object.assign(canvas.style, {
            position: "absolute",
            top: "0",
            left: "0",
            display: "block",
            pointerEvents: "none",
        });

        wrapper.appendChild(canvas);
        map.getContainer().appendChild(wrapper);

        this._wrapper = wrapper;
        this._canvas  = canvas;

        this._drawBound = this._draw.bind(this);
        map.on("move zoom viewreset resize", this._drawBound);
        this._draw();
    },

    onRemove(map) {
        map.off("move zoom viewreset resize", this._drawBound);
        this._wrapper?.remove();
        this._wrapper = null;
        this._canvas  = null;
    },

    setMetersPerPixel(mpp) {
        this._mpp = mpp;
        if (this._map) this._draw();
    },

    _draw() {
        const map    = this._map;
        const canvas = this._canvas;
        if (!map || !canvas || !this._mpp || this._mpp <= 0) return;

        const size = map.getSize();
        canvas.width  = size.x;
        canvas.height = size.y;

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, size.x, size.y);

        const mpp = this._mpp;

        // Determine the spacing: how many metres per screen pixel right now?
        const centre = map.getCenter();
        const p0 = map.latLngToContainerPoint([centre.lat, centre.lng]);
        const p1 = map.latLngToContainerPoint([centre.lat, centre.lng + 100]);
        const screenPxPerImagePx = Math.max(0.0001, Math.abs(p1.x - p0.x) / 100);
        const metersPerScreenPx  = mpp / screenPxPerImagePx;
        const targetM = 150 * metersPerScreenPx; // aim for ~150px between lines

        const niceSpacings = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
        const gridM = niceSpacings.reduce((best, s) =>
            Math.abs(Math.log(s) - Math.log(targetM)) <
            Math.abs(Math.log(best) - Math.log(targetM)) ? s : best
        );
        const gridPx = gridM / mpp; // image pixels per grid cell

        const bounds = map.getBounds();

        // Grid lines aligned to multiples of gridPx from the image origin
        const startX   = Math.floor(bounds.getWest()  / gridPx) * gridPx;
        const startLat = Math.floor(bounds.getSouth() / gridPx) * gridPx;

        ctx.strokeStyle = "rgba(126, 200, 227, 0.32)";
        ctx.lineWidth   = 1;

        // Vertical lines (constant lng = constant image x)
        for (let x = startX; x <= bounds.getEast() + gridPx; x += gridPx) {
            const a = map.latLngToContainerPoint([bounds.getSouth() - 100, x]);
            const b = map.latLngToContainerPoint([bounds.getNorth() + 100, x]);
            ctx.beginPath();
            ctx.moveTo(Math.round(a.x), 0);
            ctx.lineTo(Math.round(b.x), size.y);
            ctx.stroke();
        }

        // Horizontal lines (constant lat = constant image y through CRS.Simple)
        for (let lat = startLat; lat <= bounds.getNorth() + gridPx; lat += gridPx) {
            const a = map.latLngToContainerPoint([lat, bounds.getWest()  - 100]);
            ctx.beginPath();
            ctx.moveTo(0,      Math.round(a.y));
            ctx.lineTo(size.x, Math.round(a.y));
            ctx.stroke();
        }

        // Label the grid spacing once, top-left corner
        const labelText = gridM >= 1000 ? `${gridM / 1000} km grid` : `${gridM} m grid`;
        ctx.fillStyle = "rgba(126, 200, 227, 0.85)";
        ctx.font      = `11px "JetBrains Mono", monospace`;
        ctx.textBaseline = "top";
        ctx.textAlign    = "left";
        ctx.fillText(labelText, 8, 8);
    },
});

// ----------------------------------------------------------------
// Layer management
// ----------------------------------------------------------------
function addLayerIfMissing(layer) {
    if (state.map && layer && !state.map.hasLayer(layer)) {
        layer.addTo(state.map);
    }
}

function removeLayerIfPresent(layer) {
    if (state.map && layer && state.map.hasLayer(layer)) {
        state.map.removeLayer(layer);
    }
}

function clearDestinationLayers() {
    for (const layer of Object.values(state.layers)) {
        removeLayerIfPresent(layer);
    }
    state.layers = {
        regionalImagery: null,
        detailImagery:   null,
        cost:            null,
        routeCasing:     null,
        routeLine:       null,
        endpoints:       null,
        destinationPins: null,
        basecampPin:     null,
    };
}

// ----------------------------------------------------------------
// Destination pin creation
// ----------------------------------------------------------------
function createDestinationPins() {
    const { mission, scienceSites, selectedDestination } = state;
    const overview = mission.overview;

    const scaleX = overview.width  / mission.raster.width;
    const scaleY = overview.height / mission.raster.height;

    const markers = [];

    for (const site of scienceSites.destinations) {
        const pixel = site.raster_pixel;
        if (!pixel || !Number.isFinite(pixel.x) || !Number.isFinite(pixel.y)) {
            console.warn(`Missing raster pixel for ${site.id}`);
            continue;
        }

        const x = pixel.x * scaleX;
        const y = pixel.y * scaleY;

        const isSelected = selectedDestination && selectedDestination.id === site.id;

        const marker = L.marker(
            [overview.height - y, x],
            {
                icon: L.divIcon({
                    className: "science-site-marker",
                    html: `
                        <div class="science-pin${isSelected ? " is-selected" : ""}">
                            <span class="science-pin-dot"></span>
                            <span class="science-pin-label">${escapeHTML(site.name)}</span>
                        </div>
                    `,
                    iconSize:   [0, 0],
                    iconAnchor: [0, 0],
                }),
                title:    site.name,
                keyboard: true,
                alt:      site.name,
            }
        );

        marker.on("click", () => openDestinationModal(site));
        markers.push(marker);
    }

    return L.layerGroup(markers);
}

// ----------------------------------------------------------------
// Basecamp marker
// ----------------------------------------------------------------
function createBasecampPin() {
    const latLng = getBasecampLatLng();
    if (!latLng) return null;

    return L.marker(latLng, {
        icon: L.divIcon({
            className: "basecamp-marker",
            html: `
                <div class="basecamp-pin">
                    <span class="basecamp-pin-diamond"></span>
                    <span class="basecamp-pin-label">Basecamp</span>
                </div>
            `,
            iconSize:   [0, 0],
            iconAnchor: [0, 0],
        }),
        title:       "Rover basecamp (candidate 4)",
        keyboard:    false,
        interactive: false,
        zIndexOffset: -100,
    });
}

// ----------------------------------------------------------------
// Map layer creation
// ----------------------------------------------------------------
function createMapLayers() {
    const { mission, route } = state;
    const overview = mission.overview;
    const detail   = mission.detail;

    const overviewBounds = imageBounds(overview.width, overview.height);
    const detailBounds   = imageBounds(detail.width,   detail.height);

    state.layers.regionalImagery = L.imageOverlay(
        imageURL(overview.file), overviewBounds,
        { opacity: 1, interactive: false, className: "surface-image" }
    );

    state.layers.detailImagery = L.imageOverlay(
        imageURL(detail.surface_file), detailBounds,
        { opacity: 1, interactive: false, className: "surface-image" }
    );

    state.layers.cost = L.imageOverlay(
        imageURL(detail.cost_file), detailBounds,
        { opacity: DEFAULT_COST_OPACITY, interactive: false, className: "cost-image" }
    );

    const routeLatLngs = routeToLatLng(route.coordinates, detail.height);

    state.layers.routeCasing = L.polyline(routeLatLngs, {
        color:      "#071019",
        weight:     7,
        opacity:    0.9,
        lineCap:    "round",
        lineJoin:   "round",
        interactive: false,
    });

    state.layers.routeLine = L.polyline(routeLatLngs, {
        color:       "#7ec8e3",
        weight:      3.2,
        opacity:     1,
        lineCap:    "round",
        lineJoin:   "round",
        interactive: false,
    });

    const start  = routeLatLngs[0];
    const finish = routeLatLngs[routeLatLngs.length - 1];

    const startMarker = L.marker(start, {
        icon: L.divIcon({
            className: "basecamp-marker",
            html: `
                <div class="basecamp-pin">
                    <span class="basecamp-pin-diamond"></span>
                    <span class="basecamp-pin-label">Basecamp</span>
                </div>
            `,
            iconSize:   [0, 0],
            iconAnchor: [0, 0],
        }),
        title:        "Rover basecamp",
        keyboard:     false,
        interactive:  false,
        zIndexOffset: -100,
    });

    const finishMarker = L.circleMarker(finish, {
        radius:      6,
        color:       "#071019",
        weight:      2,
        fillColor:   "#d4895a",
        fillOpacity: 1,
    }).bindTooltip("Destination", { direction: "top", offset: [0, -5] });

    state.layers.endpoints     = L.layerGroup([startMarker, finishMarker]);
    state.layers.destinationPins = createDestinationPins();
    state.layers.basecampPin   = createBasecampPin();
}

// ----------------------------------------------------------------
// Viewport management
// ----------------------------------------------------------------
function fitMapToImage(map, bounds, imageWidth, imageHeight) {
    map.invalidateSize({ pan: false });
    map.setMaxBounds(bounds);
    map.fitBounds(bounds, { animate: false, padding: [MAP_PADDING, MAP_PADDING] });

    const baseZoom    = map.getZoom();
    const viewport    = map.getSize();
    const availW      = Math.max(1, viewport.x - MAP_PADDING * 2);
    const availH      = Math.max(1, viewport.y - MAP_PADDING * 2);
    const viewAspect  = availW / availH;
    const imageAspect = imageWidth / imageHeight;
    const zoomBoost   = Math.max(0, Math.log2(viewAspect / imageAspect));
    const targetZoom  = Math.min(map.getMaxZoom(), baseZoom + zoomBoost);

    map.setMinZoom(baseZoom - 0.5);
    map.setZoom(targetZoom, { animate: false });
    map.invalidateSize({ pan: false });
}

/**
 * Fit the regional view viewport so the science destinations and
 * basecamp are immediately visible — not the geometric image centre.
 */
function fitToScienceArea() {
    const { map, mission, scienceSites } = state;
    if (!map || !mission) return;

    const { overview, raster } = mission;
    const overviewBounds = imageBounds(overview.width, overview.height);

    // Establish max panning bounds and min zoom from full image
    map.setMaxBounds(overviewBounds);
    map.invalidateSize({ pan: false });
    map.fitBounds(overviewBounds, { animate: false, padding: [MAP_PADDING, MAP_PADDING] });
    const baseZoom = map.getZoom();
    map.setMinZoom(baseZoom - 0.5);

    if (!scienceSites?.destinations?.length) return;

    // Collect all feature points in Leaflet CRS.Simple [lat, lng] space
    const points = [];

    for (const site of scienceSites.destinations) {
        const px = site.raster_pixel;
        if (!px || !Number.isFinite(px.x) || !Number.isFinite(px.y)) continue;
        const ox = px.x * (overview.width  / raster.width);
        const oy = px.y * (overview.height / raster.height);
        points.push([overview.height - oy, ox]);
    }

    const basecampLL = getBasecampLatLng();
    if (basecampLL) points.push(basecampLL);

    if (points.length === 0) return;

    const featureBounds = L.latLngBounds(points);

    // Fit to science area with generous padding so nothing sits under floating panels
    map.fitBounds(featureBounds, {
        animate: false,
        paddingTopLeft: [300, 80],
        paddingBottomRight: [300, 80],
    });
}

function fitCurrentView() {
    if (!state.map || !state.mission) return;

    if (state.activeView === "regional") {
        // Always focus on the science area in regional mode
        fitToScienceArea();
    } else {
        const { detail } = state.mission;
        const bounds = imageBounds(detail.width, detail.height);
        fitMapToImage(state.map, bounds, detail.width, detail.height);
    }
}

// ----------------------------------------------------------------
// Layer visibility management
// ----------------------------------------------------------------
function updateLayerControls() {
    if (!state.map) return;

    const regional         = state.activeView === "regional";
    const imageryVisible   = byId("layer-imagery").checked;
    const basecampVisible  = byId("layer-basecamp").checked;
    const pinsVisible      = byId("layer-footprint").checked;
    const costVisible      = byId("layer-cost").checked;
    const routeVisible     = byId("layer-route").checked;

    const costControl      = byId("layer-cost");
    const routeControl     = byId("layer-route");
    const opacityControl   = byId("cost-opacity");
    const opacityPanel     = byId("cost-opacity-control");

    costControl.disabled   = regional;
    routeControl.disabled  = regional;
    opacityControl.disabled = regional || !costVisible;

    // Disable the eye buttons for cost/route in regional view
    const costEye  = byId("eye-layer-cost");
    const routeEye = byId("eye-layer-route");
    if (costEye)  { costEye.disabled  = regional; }
    if (routeEye) { routeEye.disabled = regional; }

    costControl.closest(".layer-row").classList.toggle("is-disabled", regional);
    routeControl.closest(".layer-row").classList.toggle("is-disabled", regional);
    opacityPanel.classList.toggle("is-disabled", regional || !costVisible);

    if (regional) {
        if (imageryVisible)  addLayerIfMissing(state.layers.regionalImagery);
        else removeLayerIfPresent(state.layers.regionalImagery);

        if (pinsVisible)     addLayerIfMissing(state.layers.destinationPins);
        else removeLayerIfPresent(state.layers.destinationPins);

        if (basecampVisible && state.layers.basecampPin)
            addLayerIfMissing(state.layers.basecampPin);
        else removeLayerIfPresent(state.layers.basecampPin);

        removeLayerIfPresent(state.layers.detailImagery);
        removeLayerIfPresent(state.layers.cost);
        removeLayerIfPresent(state.layers.routeCasing);
        removeLayerIfPresent(state.layers.routeLine);
        removeLayerIfPresent(state.layers.endpoints);

    } else {
        if (imageryVisible)  addLayerIfMissing(state.layers.detailImagery);
        else removeLayerIfPresent(state.layers.detailImagery);

        if (costVisible)     addLayerIfMissing(state.layers.cost);
        else removeLayerIfPresent(state.layers.cost);

        if (routeVisible) {
            addLayerIfMissing(state.layers.routeCasing);
            addLayerIfMissing(state.layers.routeLine);
            addLayerIfMissing(state.layers.endpoints);
        } else {
            removeLayerIfPresent(state.layers.routeCasing);
            removeLayerIfPresent(state.layers.routeLine);
            removeLayerIfPresent(state.layers.endpoints);
        }

        removeLayerIfPresent(state.layers.regionalImagery);
        removeLayerIfPresent(state.layers.destinationPins);
        removeLayerIfPresent(state.layers.basecampPin);
    }

    // Cost legend visibility
    const costLegendVisible = !regional && costVisible;
    byId("cost-legend").hidden = !costLegendVisible;

    // Map feature legend: show route row only in detail view, basecamp/dest only in regional
    const routeRow     = byId("legend-route-row");
    const basecampRow  = byId("legend-basecamp-row");
    if (routeRow)    routeRow.style.display    = regional ? "none" : "";
    if (basecampRow) basecampRow.style.display = regional ? "" : "none";

    // Status readout
    if (!regional && costVisible)        setRouteStatus("Cost layer active");
    else if (!regional && routeVisible)  setRouteStatus("Route overlay active");
    else if (regional)                   setRouteStatus("Regional view");
    else                                 setRouteStatus("Mission area view");
}

// ----------------------------------------------------------------
// View switching
// ----------------------------------------------------------------
function updateViewChrome(view) {
    const isRegional     = view === "regional";
    const regionalButton = byId("view-regional");
    const detailButton   = byId("view-detail");

    regionalButton.classList.toggle("active", isRegional);
    regionalButton.setAttribute("aria-pressed", String(isRegional));
    detailButton.classList.toggle("active", !isRegional);
    detailButton.setAttribute("aria-pressed", String(!isRegional));

    const mapTitleEl = byId("map-title");
    if (mapTitleEl) mapTitleEl.textContent = isRegional ? "Regional overview" : "Mission area detail";
    const mapViewLabelEl = byId("map-view-label");
    if (mapViewLabelEl) mapViewLabelEl.textContent = isRegional ? "Regional pixels" : "Mission area pixels";
    byId("map-hint").hidden         = !isRegional;
}

function switchView(view) {
    if (!state.map || !["regional", "detail"].includes(view)) return;
    if (state.activeView === view) return;

    state.activeView = view;

    // Update grid scale for the new view
    if (state.gridLayer) {
        const mpp = view === "regional"
            ? state.metersPerPxOverview
            : state.metersPerPxDetail;
        if (mpp) state.gridLayer.setMetersPerPixel(mpp);
    }

    updateViewChrome(view);
    updateLayerControls();
    fitCurrentView();

    byId("map-coordinate-readout").textContent = "Move cursor over map for coordinates";
    setRouteStatus(view === "regional" ? "Regional view" : "Mission area view");
}

// ----------------------------------------------------------------
// Status helpers
// ----------------------------------------------------------------
function setStatus(status) {
    const el = document.getElementById("system-status-value");
    if (el) el.textContent = status;
    setRouteStatus(status);
}

function setRouteStatus(status) {
    const el = byId("routebar-status");
    if (el) el.textContent = status;
}

// ----------------------------------------------------------------
// Destination selector
// ----------------------------------------------------------------
function populateDestinationSelector(destinations, selectedId) {
    const selector = byId("destination-select");
    selector.replaceChildren();
    for (const dest of destinations) {
        const opt = document.createElement("option");
        opt.value       = dest.id;
        opt.textContent = dest.name;
        selector.append(opt);
    }
    selector.value    = selectedId;
    selector.disabled = false;
}

// ----------------------------------------------------------------
// Destination data loading
// ----------------------------------------------------------------
async function loadDestination(destination) {
    const [mission, route] = await Promise.all([
        loadJSON(destination.mission_file),
        loadJSON(destination.route_file),
    ]);

    validateMissionData(mission, route);

    await Promise.all([
        verifyImage(imageURL(mission.detail.surface_file), mission.detail.width, mission.detail.height),
        verifyImage(imageURL(mission.detail.cost_file),    mission.detail.width, mission.detail.height),
    ]);

    state.mission             = mission;
    state.route               = route;
    state.selectedDestination = destination;

    // Compute metres-per-pixel for both views once per mission load
    state.metersPerPxDetail   = Math.abs(mission.raster.transform[0]);
    const ovTransform         = (() => {
        // getOverviewTransform() needs state.mission to be set, which it now is
        if (Array.isArray(mission.overview.transform) && mission.overview.transform.length >= 6) {
            return mission.overview.transform;
        }
        const sx = mission.raster.width  / mission.overview.width;
        const sy = mission.raster.height / mission.overview.height;
        const [a, b, c, d, e, f] = mission.raster.transform;
        return [a * sx, b * sy, c, d * sx, e * sy, f];
    })();
    state.metersPerPxOverview = Math.abs(ovTransform[0]);

    byId("destination-name").textContent = destination.name;
    byId("destination-select").value     = destination.id;
    byId("route-status").textContent =
        `${mission.route.cells.toLocaleString()} cells · exported`;

    if (state.map) {
        clearDestinationLayers();
        createMapLayers();
        updateLayerControls();
        fitCurrentView();

        // Update grid mpp for the current view
        if (state.gridLayer) {
            const mpp = state.activeView === "regional"
                ? state.metersPerPxOverview
                : state.metersPerPxDetail;
            if (mpp) state.gridLayer.setMetersPerPixel(mpp);
        }
    }

    populateRouteMetrics();
    renderElevationProfile();
}

function validateMissionData(mission, route) {
    const positive = (v) => Number.isFinite(v) && v > 0;

    if (!mission?.overview?.file || !positive(mission.overview.width) || !positive(mission.overview.height))
        throw new Error("Regional image metadata is missing or invalid.");

    if (!mission?.detail?.surface_file || !mission.detail.cost_file ||
        !positive(mission.detail.width) || !positive(mission.detail.height))
        throw new Error("Mission-area image metadata is missing or invalid.");

    if (!mission.raster || !positive(mission.raster.width) || !positive(mission.raster.height) ||
        !Array.isArray(mission.raster.transform) || mission.raster.transform.length < 6)
        throw new Error("Source raster metadata is missing or invalid.");

    if (!Number.isFinite(mission.detail.row_offset) || !Number.isFinite(mission.detail.column_offset) ||
        mission.detail.row_offset < 0 || mission.detail.column_offset < 0 ||
        mission.detail.row_offset    + mission.detail.height > mission.raster.height ||
        mission.detail.column_offset + mission.detail.width  > mission.raster.width)
        throw new Error("Mission-area crop offsets fall outside the source raster.");

    if (!Array.isArray(route?.coordinates) || route.coordinates.length < 2)
        throw new Error("Route data is missing or contains fewer than two points.");

    for (const [i, pt] of route.coordinates.entries()) {
        if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y))
            throw new Error(`Route coordinate ${i} is invalid.`);
        if (pt.x < 0 || pt.y < 0 || pt.x > mission.detail.width || pt.y > mission.detail.height)
            throw new Error(`Route coordinate ${i} falls outside the detail image.`);
    }

    const rm = mission.route;
    if (!rm || !Number.isFinite(rm.distance_m) || !Number.isFinite(rm.modeled_cost_m_equivalent) ||
        !Number.isFinite(rm.cells))
        throw new Error("Route summary metrics are missing or invalid.");
}

// ----------------------------------------------------------------
// Map initialisation
// ----------------------------------------------------------------
async function initializeMap() {
    if (typeof L === "undefined") {
        throw new Error("Leaflet failed to load. Check vendor/leaflet files.");
    }

    const { mission } = state;
    const overview = mission.overview;
    const detail   = mission.detail;

    await Promise.all([
        verifyImage(imageURL(overview.file),         overview.width, overview.height),
        verifyImage(imageURL(detail.surface_file),   detail.width,   detail.height),
        verifyImage(imageURL(detail.cost_file),      detail.width,   detail.height),
    ]);

    const map = L.map("map", {
        crs:                L.CRS.Simple,
        attributionControl: false,
        minZoom:            -3,
        maxZoom:            4,
        zoomSnap:           0.25,
        zoomDelta:          0.5,
        wheelPxPerZoomLevel: 100,
        maxBoundsViscosity: 0.85,
        zoomControl:        true,
    });

    state.map = map;

    // Metric scale control — mandatory per spec
    L.control.scale({
        imperial: false,
        position: "bottomright",
        maxWidth: 160,
    }).addTo(map);

    // Metric grid layer — enabled by default
    state.gridLayer = new MetricGridLayer({
        metersPerPixel: state.metersPerPxOverview || 1,
    });
    if (byId("layer-grid")?.checked) {
        state.gridLayer.addTo(map);
    }

    createMapLayers();

    state.layers.regionalImagery.addTo(map);
    state.layers.destinationPins.addTo(map);
    if (state.layers.basecampPin) state.layers.basecampPin.addTo(map);

    // Initial viewport: focus on the science area, not the image centre
    fitToScienceArea();

    map.on("mousemove", (e) => updateCoordinateReadout(e.latlng));
    map.on("mouseout",  () => {
        byId("map-coordinate-readout").textContent = "Move cursor over map for coordinates";
    });
    map.on("click", (e) => {
        const pixel = state.activeView === "regional"
            ? mapPointToImagePixel(e.latlng, overview.height)
            : mapPointToImagePixel(e.latlng, detail.height);
        console.info("Image pixel:", pixel);
    });

    map.whenReady(() => {
        map.invalidateSize();
        fitToScienceArea(); // ensure correct viewport after Leaflet finishes sizing
    });

    if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(() => {
            map.invalidateSize({ pan: false });
            fitCurrentView();
        });
        observer.observe(byId("map"));
        state.resizeObserver = observer;
    } else {
        window.addEventListener("resize", () => {
            map.invalidateSize({ pan: false });
            fitCurrentView();
        });
    }

    updateLayerControls();
}

// ----------------------------------------------------------------
// Route metrics display
// ----------------------------------------------------------------
function populateRouteMetrics() {
    const rm = state.mission.route;

    byId("metric-distance").textContent = formatDistance(rm.distance_m);
    byId("metric-cost").textContent     = formatDistance(rm.modeled_cost_m_equivalent);
    byId("metric-cells").textContent    = rm.cells.toLocaleString();
    byId("route-status").textContent    = `${rm.cells.toLocaleString()} cells · exported`;
    byId("expanded-cells").textContent  = rm.expanded.toLocaleString();
}

// ----------------------------------------------------------------
// Elevation profile
// ----------------------------------------------------------------
function renderElevationProfile() {
    const container = byId("elevation-profile");

    const profile =
        state.route.elevation_profile ||
        state.mission.route.elevation_profile;

    if (!Array.isArray(profile)) {
        container.className = "empty-state";
        container.innerHTML = `<span class="empty-icon" aria-hidden="true">⌁</span><span>Elevation profile unavailable.</span>`;
        return;
    }

    const samples = profile.filter((p) =>
        Number.isFinite(p.distance_m) && Number.isFinite(p.elevation_m)
    );

    if (samples.length < 2) {
        container.className = "empty-state";
        container.innerHTML = `<span>Insufficient DEM samples for profile.</span>`;
        return;
    }

    const W = 320, H = 108, pX = 8, pY = 8;
    const dists = samples.map((p) => p.distance_m);
    const elevs = samples.map((p) => p.elevation_m);
    const minD  = Math.min(...dists), maxD = Math.max(...dists);
    const minE  = Math.min(...elevs), maxE = Math.max(...elevs);
    const dSpan = maxD - minD || 1;
    const eSpan = maxE - minE || 1;

    const pts = samples.map((p) => {
        const x = pX + ((p.distance_m  - minD) / dSpan) * (W - 2 * pX);
        const y = H - pY - ((p.elevation_m - minE) / eSpan) * (H - 2 * pY);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const polyline = pts.join(" ");
    const area     = `${pX},${H - pY} ${polyline} ${W - pX},${H - pY}`;
    const gid      = "ares-elev-gradient";

    container.className = "elevation-profile";
    container.innerHTML = `
        <div class="profile-range">
            <span>${maxE.toFixed(1)} m</span>
            <span>${minE.toFixed(1)} m</span>
        </div>
        <svg viewBox="0 0 ${W} ${H}" role="img"
             aria-label="Elevation profile from ${formatDistance(minD)} to ${formatDistance(maxD)}">
            <defs>
                <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stop-color="#7ec8e3" stop-opacity="0.22"/>
                    <stop offset="100%" stop-color="#7ec8e3" stop-opacity="0.02"/>
                </linearGradient>
            </defs>
            <line x1="${pX}" y1="${H - pY}" x2="${W - pX}" y2="${H - pY}"
                  stroke="#1f2d3d" stroke-width="1"/>
            <polygon points="${area}" fill="url(#${gid})"/>
            <polyline points="${polyline}" fill="none" stroke="#7ec8e3"
                      stroke-width="2" vector-effect="non-scaling-stroke"
                      stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="profile-distance">
            <span>${formatDistance(minD)}</span>
            <span>${formatDistance(maxD)}</span>
        </div>
        <div class="profile-stat">
            <span>Relief</span>
            <strong>${(maxE - minE).toFixed(1)} m</strong>
            <span>${samples.length} samples</span>
        </div>
    `;
}

// ----------------------------------------------------------------
// Science destination modal
// ----------------------------------------------------------------
function openDestinationModal(site) {
    const dialog  = byId("destination-modal");
    const content = byId("destination-modal-content");
    if (!dialog || !content) { console.error("Modal markup missing."); return; }

    const coordinates  = site.coordinates || {};
    const evidence     = Array.isArray(site.evidence)     ? site.evidence     : [];
    const observations = Array.isArray(site.observations) ? site.observations : [];

    // Evidence blocks
    const evidenceHTML = evidence.length
        ? evidence.map((item) => `
            <article class="modal-evidence">
                <div class="modal-eyebrow">${escapeHTML(item.evidence_class || item.type || "Evidence")}</div>
                <p>${escapeHTML(item.claim)}</p>
                ${item.spatial_relation  ? `<p class="modal-secondary">${escapeHTML(item.spatial_relation)}</p>` : ""}
                ${item.confidence        ? `<p class="modal-secondary"><strong>Confidence:</strong> ${escapeHTML(item.confidence)}</p>` : ""}
            </article>`).join("")
        : `<p class="modal-secondary">No evidence records available.</p>`;

    // Observation blocks
    const observationsHTML = observations.length
        ? observations.map((item) => {
            const imgPath  = item.web_asset ? imageURL(item.web_asset) : null;
            const srcURL   = safeExternalURL(item.url);
            return `
                <article class="modal-observation">
                    ${imgPath ? `
                        <a class="modal-observation-image" href="${escapeHTML(imgPath)}"
                           target="_blank" rel="noopener noreferrer">
                            <img src="${escapeHTML(imgPath)}"
                                 alt="${escapeHTML(item.asset_id || "Science observation")}"
                                 loading="lazy">
                        </a>` : ""}
                    <div class="modal-observation-info">
                        <div class="modal-eyebrow">
                            ${escapeHTML(item.asset_id || "Observation")}
                            · ${escapeHTML(item.instrument || "Instrument unspecified")}
                        </div>
                        <p>${escapeHTML(item.description)}</p>
                        ${item.relation_to_waypoint ? `<p class="modal-secondary">${escapeHTML(item.relation_to_waypoint)}</p>` : ""}
                        ${srcURL ? `<a href="${escapeHTML(srcURL)}" target="_blank" rel="noopener noreferrer">View NASA source ↗</a>` : ""}
                    </div>
                </article>`;
        }).join("")
        : `<p class="modal-secondary">No image observations catalogued for this site.</p>`;

    // Source references
    const sourceIDs      = Array.isArray(site.sources)                           ? site.sources                           : [];
    const sourceRegistry = Array.isArray(state.scienceSites?.source_registry)    ? state.scienceSites.source_registry    : [];
    const sourceById     = new Map(sourceRegistry.map((s) => [s.id, s]));

    const sourcesHTML = sourceIDs.length
        ? sourceIDs.map((id) => {
            const source = sourceById.get(id);
            if (!source) return `
                <article class="modal-source-entry">
                    <span class="modal-source-id">${escapeHTML(id)}</span>
                    <p class="modal-source-missing">Source metadata not available.</p>
                </article>`;

            const srcURL   = safeExternalURL(source.url);
            const metadata = [source.year, source.publisher].filter(Boolean).join(" · ");
            return `
                <article class="modal-source-entry">
                    <div class="modal-source-heading">
                        <span class="modal-source-id">${escapeHTML(source.id)}</span>
                        ${metadata ? `<span class="modal-source-metadata">${escapeHTML(metadata)}</span>` : ""}
                    </div>
                    <p class="modal-source-title">${escapeHTML(source.title || source.id)}</p>
                    ${source.role ? `<p class="modal-source-role">${escapeHTML(source.role)}</p>` : ""}
                    ${srcURL ? `<a href="${escapeHTML(srcURL)}" target="_blank" rel="noopener noreferrer">Open source ↗</a>` : ""}
                </article>`;
        }).join("")
        : `<p class="modal-secondary">No source references listed.</p>`;

    // Populate sticky header elements
    byId("modal-site-id").textContent = site.id || "Target";
    byId("modal-site-title").textContent = site.name;
    byId("modal-site-geo").textContent = site.geological_unit || "Geological unit unspecified";
    byId("modal-site-coords").textContent = (Number.isFinite(coordinates.longitude_deg) && Number.isFinite(coordinates.latitude_deg))
        ? `${coordinates.longitude_deg.toFixed(3)}° E · ${coordinates.latitude_deg.toFixed(3)}° N`
        : "Coordinates unavailable";
    byId("modal-route-button").dataset.destinationId = site.id;

    content.innerHTML = `
        <section class="modal-section">
            <h3>Site overview</h3>
            <p>${escapeHTML(site.summary)}</p>
        </section>

        <section class="modal-section">
            <h3>Scientific significance</h3>
            <p>${escapeHTML(site.scientific_significance)}</p>
        </section>

        <section class="modal-section">
            <h3>Exploration objective</h3>
            <p>${escapeHTML(site.science_objective)}</p>
        </section>

        <section class="modal-section">
            <h3>Geological evidence</h3>
            ${evidenceHTML}
        </section>

        <section class="modal-section">
            <h3>Imagery and observations</h3>
            <div class="modal-observations">${observationsHTML}</div>
        </section>

        <section class="modal-section modal-section-sources">
            <details class="modal-sources-toggle">
                <summary class="modal-sources-summary">
                    <div class="modal-sources-summary-title">
                        <h3>Research sources</h3>
                        <span class="modal-source-count">${sourceIDs.length}</span>
                    </div>
                    <svg class="modal-toggle-chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <polyline points="4 6 8 10 12 6"/>
                    </svg>
                </summary>
                <div class="modal-source-list">${sourcesHTML}</div>
            </details>
        </section>
    `;

    // Graceful image fallback
    content.querySelectorAll(".modal-observation-image img").forEach((img) => {
        img.addEventListener("error", () => {
            const link = img.closest(".modal-observation-image");
            if (link) {
                link.hidden = true;
                const note = document.createElement("p");
                note.className = "modal-secondary";
                note.textContent = "Local image asset unavailable. Use the source link below.";
                link.insertAdjacentElement("afterend", note);
            }
        }, { once: true });
    });

    byId("modal-route-button").dataset.destinationId = site.id;
    if (!dialog.open) dialog.showModal();
}

async function routeToDestination(destinationId) {
    const destination = state.destinations.find((d) => d.id === destinationId);
    if (!destination) throw new Error(`No precomputed route for ${destinationId}.`);

    const selector   = byId("destination-select");
    const routeBtn   = byId("modal-route-button");
    selector.disabled = true;
    routeBtn.disabled = true;
    setStatus("Loading route");

    try {
        await loadDestination(destination);
        switchView("detail");
        byId("destination-modal").close();
        setStatus("Ready");
    } catch (error) {
        console.error("Could not load destination route:", error);
        selector.value = state.selectedDestination?.id || "";
        setRouteStatus(`Route load failed: ${error.message}`);
        setStatus("Data error");
    } finally {
        selector.disabled = false;
        routeBtn.disabled = false;
    }
}

// ----------------------------------------------------------------
// Controls
// ----------------------------------------------------------------
function connectControls() {
    byId("view-regional").addEventListener("click", () => switchView("regional"));
    byId("view-detail").addEventListener("click",   () => switchView("detail"));

    byId("destination-select").addEventListener("change", async (e) => {
        const destination = state.destinations.find((d) => d.id === e.target.value);
        if (!destination) return;

        const selector    = byId("destination-select");
        selector.disabled = true;
        setStatus("Loading destination");

        try {
            await loadDestination(destination);
            setStatus("Ready");
        } catch (error) {
            console.error("Could not load destination:", error);
            selector.value = state.selectedDestination?.id || "";
            setRouteStatus(`Load failed: ${error.message}`);
            setStatus("Data error");
        } finally {
            selector.disabled = false;
        }
    });

    // Layer eye-icon toggles — sync hidden checkbox and update layers
    function setupEyeToggle(layerId) {
        const eyeBtn  = byId("eye-" + layerId);
        const checkbox = byId(layerId);
        if (!eyeBtn || !checkbox) return;

        eyeBtn.addEventListener("click", () => {
            checkbox.checked = !checkbox.checked;
            const visible = checkbox.checked;
            eyeBtn.setAttribute("aria-pressed", String(visible));
            // Swap eye-on / eye-off SVG visibility
            const eyeOn  = eyeBtn.querySelector(".eye-on");
            const eyeOff = eyeBtn.querySelector(".eye-off");
            if (eyeOn)  eyeOn.style.display  = visible ? "" : "none";
            if (eyeOff) eyeOff.style.display = visible ? "none" : "";

            // For grid, handle separately; for others call updateLayerControls
            if (layerId === "layer-grid") {
                if (visible) {
                    if (state.gridLayer && state.map) {
                        const mpp = state.activeView === "regional"
                            ? state.metersPerPxOverview
                            : state.metersPerPxDetail;
                        if (mpp) state.gridLayer.setMetersPerPixel(mpp);
                        state.gridLayer.addTo(state.map);
                    }
                } else {
                    if (state.gridLayer) removeLayerIfPresent(state.gridLayer);
                }
            } else {
                updateLayerControls();
            }
        });
    }

    for (const id of ["layer-imagery", "layer-basecamp", "layer-footprint", "layer-cost", "layer-route", "layer-grid"]) {
        setupEyeToggle(id);
    }

    // Cost opacity
    byId("cost-opacity").addEventListener("input", (e) => {
        const opacity = Number(e.target.value) / 100;
        if (state.layers.cost) state.layers.cost.setOpacity(opacity);
        byId("cost-opacity-value").textContent = `${e.target.value}%`;
    });

    // Hint / map hint click → switch to detail view
    byId("map-hint").addEventListener("click", () => switchView("detail"));

    // Modal controls
    byId("modal-close").addEventListener("click", () => byId("destination-modal").close());
    byId("modal-route-button").addEventListener("click", async () => {
        await routeToDestination(byId("modal-route-button").dataset.destinationId);
    });
    byId("destination-modal").addEventListener("click", (e) => {
        if (e.target === byId("destination-modal")) byId("destination-modal").close();
    });
}

// ----------------------------------------------------------------
// Initialisation entry point
// ----------------------------------------------------------------
async function initializeARES() {
    try {
        setStatus("Loading data");

        const [destinationIndex, scienceCatalogue] = await Promise.all([
            loadJSON("destinations.json"),
            loadJSON("science-sites.json"),
        ]);

        if (!Array.isArray(destinationIndex.destinations) || destinationIndex.destinations.length === 0)
            throw new Error("Destination index is missing or empty.");

        if (!Array.isArray(scienceCatalogue.destinations))
            throw new Error("Science catalogue is missing its destinations array.");

        state.destinations = destinationIndex.destinations;
        state.scienceSites = scienceCatalogue;

        const missingRoute = scienceCatalogue.destinations.find(
            (site) => !state.destinations.some((d) => d.id === site.id)
        );
        if (missingRoute)
            throw new Error(`No precomputed route bundle found for ${missingRoute.id}.`);

        const initialDestination = state.destinations[0];
        populateDestinationSelector(state.destinations, initialDestination.id);

        await loadDestination(initialDestination);
        await initializeMap();

        connectControls();
        updateViewChrome("regional");
        updateLayerControls();
        setStatus("Ready");

        // Expose globals for browser console debugging
        window.ARES = {
            state, map: state.map, mission: state.mission, route: state.route,
            destinations: state.destinations, scienceSites: state.scienceSites,
            selectedDestination: state.selectedDestination, layers: state.layers,
            switchView, loadDestination, openDestinationModal,
            routeToDestination, updateLayerControls, fitCurrentView, fitToScienceArea,
        };

        console.info("ARES initialized.", {
            destinationCount: state.destinations.length,
            scienceSiteCount: state.scienceSites.destinations.length,
            selectedDestination: state.selectedDestination,
            overview: state.mission.overview,
            detail: state.mission.detail,
            routePoints: state.route.coordinates.length,
            routeMetrics: state.mission.route,
            metersPerPxOverview: state.metersPerPxOverview,
            metersPerPxDetail:   state.metersPerPxDetail,
        });

    } catch (error) {
        console.error("ARES initialization failed:", error);

        const mapEl = byId("map");
        mapEl.innerHTML = `
            <div class="map-error">
                <span class="map-error-title">Map initialization failed</span>
                <span class="map-error-message"></span>
                <span class="map-error-hint">
                    Check the browser console and confirm that all data assets and
                    Leaflet vendor files are present. The application must be served
                    over HTTP — opening index.html directly blocks fetch requests.
                </span>
            </div>`;
        mapEl.querySelector(".map-error-message").textContent = error.message;
        setStatus("Init error");
    }
}

document.addEventListener("DOMContentLoaded", initializeARES);
