"use strict";

const DATA_DIRECTORY = "./data/";
const DEFAULT_COST_OPACITY = 0.72;

const state = {
    map: null,
    mission: null,
    route: null,
    activeView: "regional",
    layers: {
        regionalImagery: null,
        detailImagery: null,
        cost: null,
        routeCasing: null,
        routeLine: null,
        endpoints: null,
        missionArea: null,
    },
};

const byId = (id) => document.getElementById(id);

async function loadJSON(filename) {
    const response = await fetch(`${DATA_DIRECTORY}${filename}`);

    if (!response.ok) {
        throw new Error(`Failed to load ${filename}: HTTP ${response.status}`);
    }

    return response.json();
}

function setStatus(status) {
    const statusElement = document.querySelector(".status-value");
    if (statusElement) statusElement.textContent = status;
    setRouteStatus(status);
}

function setRouteStatus(status) {
    const routeStatusElement = byId("routebar-status");
    if (routeStatusElement) routeStatusElement.textContent = status;
}

function validateMissionData(mission, route) {
    const positive = (value) => Number.isFinite(value) && value > 0;

    if (
        !mission?.overview?.file ||
        !positive(mission.overview.width) ||
        !positive(mission.overview.height)
    ) {
        throw new Error("Regional image metadata is missing or invalid.");
    }

    if (
        !mission?.detail?.surface_file ||
        !mission.detail.cost_file ||
        !positive(mission.detail.width) ||
        !positive(mission.detail.height)
    ) {
        throw new Error("Mission-area image metadata is missing or invalid.");
    }

    if (
        !mission.raster ||
        !positive(mission.raster.width) ||
        !positive(mission.raster.height) ||
        !Array.isArray(mission.raster.transform) ||
        mission.raster.transform.length < 6
    ) {
        throw new Error("Source raster metadata is missing or invalid.");
    }

    if (
        !Number.isFinite(mission.detail.row_offset) ||
        !Number.isFinite(mission.detail.column_offset) ||
        mission.detail.row_offset < 0 ||
        mission.detail.column_offset < 0 ||
        mission.detail.row_offset + mission.detail.height > mission.raster.height ||
        mission.detail.column_offset + mission.detail.width > mission.raster.width
    ) {
        throw new Error("Mission-area crop offsets fall outside the source raster.");
    }

    if (!Array.isArray(route?.coordinates) || route.coordinates.length < 2) {
        throw new Error("Route data is missing or contains fewer than two points.");
    }

    for (const [index, point] of route.coordinates.entries()) {
        if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
            throw new Error(`Route coordinate ${index} is invalid.`);
        }

        if (
            point.x < 0 ||
            point.y < 0 ||
            point.x > mission.detail.width ||
            point.y > mission.detail.height
        ) {
            throw new Error(`Route coordinate ${index} falls outside the detail image.`);
        }
    }

    const routeMetrics = mission.route;
    if (
        !routeMetrics ||
        !Number.isFinite(routeMetrics.distance_m) ||
        !Number.isFinite(routeMetrics.modeled_cost_m_equivalent) ||
        !Number.isFinite(routeMetrics.cells)
    ) {
        throw new Error("Route summary metrics are missing or invalid.");
    }
}

function imageBounds(width, height) {
    return [[0, 0], [height, width]];
}

function routeToLatLng(coordinates, imageHeight) {
    // Raster pixels start at the top-left. CRS.Simple's Y axis increases upward.
    return coordinates.map((point) => [imageHeight - point.y, point.x]);
}

function imageURL(filename) {
    return new URL(filename, new URL(DATA_DIRECTORY, window.location.href)).href;
}

async function verifyImage(url, expectedWidth, expectedHeight) {
    const image = new Image();
    image.src = url;

    try {
        await image.decode();
    } catch {
        throw new Error(`Could not decode map asset: ${url}`);
    }

    if (
        image.naturalWidth !== expectedWidth ||
        image.naturalHeight !== expectedHeight
    ) {
        throw new Error(
            `Image dimensions mismatch. Expected ${expectedWidth}×${expectedHeight}; ` +
            `received ${image.naturalWidth}×${image.naturalHeight}.`
        );
    }
}

function getOverviewTransform() {
    const { mission } = state;
    const overview = mission.overview;
    const raster = mission.raster;

    // overview is a resized full-raster image. This fallback also supports
    // metadata generated before overview.transform was added to the exporter.
    if (Array.isArray(overview.transform) && overview.transform.length >= 6) {
        return overview.transform;
    }

    const scaleX = raster.width / overview.width;
    const scaleY = raster.height / overview.height;
    const [a, b, c, d, e, f] = raster.transform;

    return [
        a * scaleX,
        b * scaleY,
        c,
        d * scaleX,
        e * scaleY,
        f,
    ];
}

function pixelToMars(x, y, transform) {
    const [a, b, c, d, e, f] = transform;
    return {
        x: a * x + b * y + c,
        y: d * x + e * y + f,
    };
}

function mapPointToImagePixel(latlng, imageHeight) {
    return {
        x: latlng.lng,
        y: imageHeight - latlng.lat,
    };
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
        transform = getOverviewTransform();
        rasterPixel = pixel;
    } else {
        const detail = mission.detail;
        rasterPixel = {
            x: pixel.x + detail.column_offset,
            y: pixel.y + detail.row_offset,
        };
        transform = mission.raster.transform;
    }

    const mars = pixelToMars(rasterPixel.x, rasterPixel.y, transform);

    output.textContent =
        `E ${mars.x.toFixed(1)} m · N ${mars.y.toFixed(1)} m`;
}

function updateViewChrome(view) {
    const isRegional = view === "regional";
    const regionalButton = byId("view-regional");
    const detailButton = byId("view-detail");

    regionalButton.classList.toggle("active", isRegional);
    regionalButton.setAttribute("aria-pressed", String(isRegional));
    detailButton.classList.toggle("active", !isRegional);
    detailButton.setAttribute("aria-pressed", String(!isRegional));

    byId("map-title").textContent = isRegional
        ? "Regional overview"
        : "Mission area detail";

    byId("map-view-label").textContent = isRegional
        ? "REGIONAL IMAGE PIXELS"
        : "MISSION AREA IMAGE PIXELS";

    byId("map-hint").hidden = !isRegional;
    byId("cost-legend").hidden =
        !(view === "detail" && byId("layer-cost").checked);
}

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

function updateLayerControls() {
    if (!state.map) return;

    const regional = state.activeView === "regional";
    const imageryVisible = byId("layer-imagery").checked;
    const footprintVisible = byId("layer-footprint").checked;
    const costVisible = byId("layer-cost").checked;
    const routeVisible = byId("layer-route").checked;

    const costControl = byId("layer-cost");
    const routeControl = byId("layer-route");
    const opacityControl = byId("cost-opacity");
    const opacityPanel = byId("cost-opacity-control");

    costControl.disabled = regional;
    routeControl.disabled = regional;
    opacityControl.disabled = regional || !costVisible;

    costControl.closest(".layer-option").classList.toggle("is-disabled", regional);
    routeControl.closest(".layer-option").classList.toggle("is-disabled", regional);
    opacityPanel.classList.toggle("is-disabled", regional || !costVisible);

    if (regional) {
        if (imageryVisible) addLayerIfMissing(state.layers.regionalImagery);
        else removeLayerIfPresent(state.layers.regionalImagery);

        if (footprintVisible) addLayerIfMissing(state.layers.missionArea);
        else removeLayerIfPresent(state.layers.missionArea);

        removeLayerIfPresent(state.layers.detailImagery);
        removeLayerIfPresent(state.layers.cost);
        removeLayerIfPresent(state.layers.routeCasing);
        removeLayerIfPresent(state.layers.routeLine);
        removeLayerIfPresent(state.layers.endpoints);
    } else {
        if (imageryVisible) addLayerIfMissing(state.layers.detailImagery);
        else removeLayerIfPresent(state.layers.detailImagery);

        removeLayerIfPresent(state.layers.regionalImagery);
        removeLayerIfPresent(state.layers.missionArea);

        if (costVisible) addLayerIfMissing(state.layers.cost);
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
    }

    byId("cost-legend").hidden = !(state.activeView === "detail" && costVisible);
    if (state.activeView === "detail" && costVisible) {
        setRouteStatus("COST LAYER ACTIVE");
    } else if (state.activeView === "detail" && routeVisible) {
        setRouteStatus("ROUTE OVERLAY ACTIVE");
    } else if (state.activeView === "regional") {
        setRouteStatus("REGIONAL VIEW");
    } else {
        setRouteStatus("MISSION AREA VIEW");
    }
}

function switchView(view) {
    if (!state.map || !["regional", "detail"].includes(view)) return;
    if (state.activeView === view) return;

    const { mission, map } = state;
    const regional = view === "regional";
    const bounds = regional
        ? imageBounds(mission.overview.width, mission.overview.height)
        : imageBounds(mission.detail.width, mission.detail.height);

    state.activeView = view;
    updateViewChrome(view);
    updateLayerControls();

    map.setMaxBounds(bounds);
    map.fitBounds(bounds, {
        animate: false,
        padding: [28, 28],
    });
    map.setMinZoom(map.getBoundsZoom(bounds, false, [28, 28]) - 0.5);
    map.invalidateSize({ pan: false });

    byId("map-coordinate-readout").textContent = regional
        ? "Move cursor over map for coordinates"
        : "Move cursor over map for coordinates";

    setRouteStatus(regional ? "REGIONAL VIEW" : "MISSION AREA VIEW");
}

function createMapLayers() {
    const { mission, route } = state;
    const overview = mission.overview;
    const detail = mission.detail;

    const overviewBounds = imageBounds(overview.width, overview.height);
    const detailBounds = imageBounds(detail.width, detail.height);

    state.layers.regionalImagery = L.imageOverlay(
        imageURL(overview.file),
        overviewBounds,
        { opacity: 1, interactive: false, className: "surface-image" }
    );

    state.layers.detailImagery = L.imageOverlay(
        imageURL(detail.surface_file),
        detailBounds,
        { opacity: 1, interactive: false, className: "surface-image" }
    );

    state.layers.cost = L.imageOverlay(
        imageURL(detail.cost_file),
        detailBounds,
        { opacity: DEFAULT_COST_OPACITY, interactive: false, className: "cost-image" }
    );

    const routeLatLngs = routeToLatLng(route.coordinates, detail.height);

    // A dark casing keeps the route legible on both bright and dark terrain.
    state.layers.routeCasing = L.polyline(routeLatLngs, {
        color: "#071019",
        weight: 7,
        opacity: 0.94,
        lineCap: "round",
        lineJoin: "round",
        interactive: false,
    });

    state.layers.routeLine = L.polyline(routeLatLngs, {
        color: "#91c5ff",
        weight: 3.2,
        opacity: 1,
        lineCap: "round",
        lineJoin: "round",
        interactive: false,
    });

    const start = routeLatLngs[0];
    const finish = routeLatLngs[routeLatLngs.length - 1];

    const startMarker = L.circleMarker(start, {
        radius: 6,
        color: "#071019",
        weight: 2.5,
        fillColor: "#79c9a4",
        fillOpacity: 1,
    }).bindTooltip("START", { direction: "top", offset: [0, -5] });

    const finishMarker = L.circleMarker(finish, {
        radius: 6,
        color: "#071019",
        weight: 2.5,
        fillColor: "#e7a45f",
        fillOpacity: 1,
    }).bindTooltip("DESTINATION", { direction: "top", offset: [0, -5] });

    state.layers.endpoints = L.layerGroup([startMarker, finishMarker]);

    const overviewScaleX = overview.width / mission.raster.width;
    const overviewScaleY = overview.height / mission.raster.height;

    const west = detail.column_offset * overviewScaleX;
    const east = (detail.column_offset + detail.width) * overviewScaleX;
    const north = detail.row_offset * overviewScaleY;
    const south = (detail.row_offset + detail.height) * overviewScaleY;

    const footprintBounds = [
        [overview.height - south, west],
        [overview.height - north, east],
    ];

    state.layers.missionArea = L.rectangle(footprintBounds, {
        color: "#e4a35f",
        weight: 2,
        opacity: 1,
        fillColor: "#e4a35f",
        fillOpacity: 0.045,
        dashArray: "7 5",
        interactive: true,
    });

    state.layers.missionArea.bindTooltip(
        "MISSION AREA · OPEN DETAIL VIEW",
        { sticky: true, direction: "top" }
    );

    state.layers.missionArea.on("click", () => switchView("detail"));
}

async function initializeMap() {
    if (typeof L === "undefined") {
        throw new Error("Leaflet failed to load. Check vendor/leaflet files.");
    }

    const { mission } = state;
    const overview = mission.overview;
    const detail = mission.detail;

    await Promise.all([
        verifyImage(imageURL(overview.file), overview.width, overview.height),
        verifyImage(imageURL(detail.surface_file), detail.width, detail.height),
        verifyImage(imageURL(detail.cost_file), detail.width, detail.height),
    ]);

    const map = L.map("map", {
        crs: L.CRS.Simple,
        attributionControl: false,
        minZoom: -3,
        maxZoom: 4,
        zoomSnap: 0.25,
        zoomDelta: 0.5,
        wheelPxPerZoomLevel: 100,
        maxBoundsViscosity: 0.85,
        zoomControl: true,
    });

    state.map = map;
    createMapLayers();

    state.layers.regionalImagery.addTo(map);
    state.layers.missionArea.addTo(map);

    const bounds = imageBounds(overview.width, overview.height);
    map.fitBounds(bounds, { animate: false, padding: [28, 28] });
    map.setMaxBounds(bounds);

    map.on("mousemove", (event) => updateCoordinateReadout(event.latlng));
    map.on("mouseout", () => {
        byId("map-coordinate-readout").textContent =
            "Move cursor over map for coordinates";
    });

    map.on("click", (event) => {
        if (state.activeView === "regional") {
            const pixel = mapPointToImagePixel(event.latlng, overview.height);
            console.info("Regional image pixel:", pixel);
        } else {
            const pixel = mapPointToImagePixel(event.latlng, detail.height);
            console.info("Mission-area image pixel:", pixel);
        }
    });

    map.whenReady(() => map.invalidateSize());

    if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(() => map.invalidateSize({ pan: false }));
        observer.observe(byId("map"));
        state.resizeObserver = observer;
    } else {
        window.addEventListener("resize", () => map.invalidateSize({ pan: false }));
    }

    updateLayerControls();
}

function populateRouteMetrics() {
    const route = state.mission.route;

    byId("metric-distance").innerHTML =
        `${route.distance_m.toFixed(1)} <small>m</small>`;

    byId("metric-cost").innerHTML =
        `${route.modeled_cost_m_equivalent.toFixed(1)} <small>m-eq</small>`;

    byId("metric-cells").textContent = route.cells.toLocaleString();
    byId("route-status").textContent = `${route.cells.toLocaleString()} CELLS · EXPORTED`;
    byId("expanded-cells").textContent = route.expanded.toLocaleString();
}

function renderElevationProfile() {
    const container = byId("elevation-profile");
    const profile = state.route.elevation_profile || state.mission.route.elevation_profile;

    if (!Array.isArray(profile)) {
        container.textContent = "Elevation profile unavailable.";
        return;
    }

    const samples = profile.filter((point) =>
        Number.isFinite(point.distance_m) &&
        Number.isFinite(point.elevation_m)
    );

    if (samples.length < 2) {
        container.className = "empty-state";
        container.textContent = "Insufficient valid DEM samples for an elevation profile.";
        return;
    }

    const width = 320;
    const height = 112;
    const padX = 8;
    const padY = 9;

    const distances = samples.map((point) => point.distance_m);
    const elevations = samples.map((point) => point.elevation_m);

    const minDistance = Math.min(...distances);
    const maxDistance = Math.max(...distances);
    const minElevation = Math.min(...elevations);
    const maxElevation = Math.max(...elevations);

    const distanceSpan = maxDistance - minDistance || 1;
    const elevationSpan = maxElevation - minElevation || 1;

    const chartPoints = samples.map((point) => {
        const x = padX +
            ((point.distance_m - minDistance) / distanceSpan) *
            (width - 2 * padX);
        const y = height - padY -
            ((point.elevation_m - minElevation) / elevationSpan) *
            (height - 2 * padY);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const polyline = chartPoints.join(" ");
    const area = `${padX},${height - padY} ${polyline} ${width - padX},${height - padY}`;
    const gradientId = "ares-profile-gradient";

    container.className = "elevation-profile";
    container.innerHTML = `
        <div class="profile-range">
            <span>${maxElevation.toFixed(1)} m</span>
            <span>${minElevation.toFixed(1)} m</span>
        </div>
        <svg viewBox="0 0 ${width} ${height}" role="img"
             aria-label="Elevation profile from ${minDistance.toFixed(1)} to ${maxDistance.toFixed(1)} metres">
            <defs>
                <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#91c5ff" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="#91c5ff" stop-opacity="0.015"/>
                </linearGradient>
            </defs>
            <line x1="${padX}" y1="${height - padY}" x2="${width - padX}" y2="${height - padY}"
                  stroke="#34404e" stroke-width="1"/>
            <polygon points="${area}" fill="url(#${gradientId})"/>
            <polyline points="${polyline}" fill="none" stroke="#91c5ff"
                      stroke-width="2.4" vector-effect="non-scaling-stroke"
                      stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="profile-distance">
            <span>${minDistance.toFixed(0)} m</span>
            <span>${maxDistance.toFixed(0)} m</span>
        </div>
        <div class="profile-stat">
            <span>RELIEF</span>
            <strong>${(maxElevation - minElevation).toFixed(1)} m</strong>
            <span>${samples.length} DEM samples</span>
        </div>
    `;
}

function connectControls() {
    byId("view-regional").addEventListener("click", () => switchView("regional"));
    byId("view-detail").addEventListener("click", () => switchView("detail"));

    for (const id of ["layer-imagery", "layer-footprint", "layer-cost", "layer-route"]) {
        byId(id).addEventListener("change", updateLayerControls);
    }

    byId("cost-opacity").addEventListener("input", (event) => {
        const opacity = Number(event.target.value) / 100;
        state.layers.cost.setOpacity(opacity);
        byId("cost-opacity-value").value = `${event.target.value}%`;
        byId("cost-opacity-value").textContent = `${event.target.value}%`;
    });

    byId("map-hint").addEventListener("click", () => switchView("detail"));
}

async function initializeARES() {
    try {
        setStatus("LOADING DATA");

        const [mission, route] = await Promise.all([
            loadJSON("mission.json"),
            loadJSON("route.json"),
        ]);

        state.mission = mission;
        state.route = route;

        validateMissionData(mission, route);
        await initializeMap();

        populateRouteMetrics();
        renderElevationProfile();
        connectControls();
        updateViewChrome("regional");
        updateLayerControls();

        setStatus("OPERATIONAL");

        window.ARES = {
            map: state.map,
            mission: state.mission,
            route: state.route,
            layers: state.layers,
            switchView,
            updateLayerControls,
        };

        console.info("ARES initialized.", {
            overview: mission.overview,
            detail: mission.detail,
            routePoints: route.coordinates.length,
            routeMetrics: mission.route,
        });
    } catch (error) {
        console.error("ARES initialization failed:", error);

        const mapElement = byId("map");
        mapElement.innerHTML = `
            <div class="map-error">
                <span class="map-error-title">MAP INITIALIZATION FAILED</span>
                <span class="map-error-message"></span>
                <span class="map-error-hint">
                    Check the browser console and confirm that the local data assets
                    and Leaflet files are present. Serve the web directory over HTTP;
                    opening index.html directly will block fetch requests.
                </span>
            </div>
        `;
        mapElement.querySelector(".map-error-message").textContent = error.message;
        setStatus("INITIALIZATION ERROR");
    }
}

document.addEventListener("DOMContentLoaded", initializeARES);
