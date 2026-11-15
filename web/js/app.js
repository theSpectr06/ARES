"use strict";

const DATA_DIRECTORY = "./data/";
const DEFAULT_COST_OPACITY = 0.72;
const MAP_PADDING = 24;

const state = {
    map: null,
    mission: null,
    route: null,
    destinations: [],
    scienceSites: null,
    selectedDestination: null,
    activeView: "regional",
    layers: {
        regionalImagery: null,
        detailImagery: null,
        cost: null,
        routeCasing: null,
        routeLine: null,
        endpoints: null,
        destinationPins: null,
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

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    })[character]);
}

function safeExternalURL(value) {
    try {
        const url = new URL(value);

        if (url.protocol === "https:" || url.protocol === "http:") {
            return url.href;
        }
    } catch {
        // Invalid URL.
    }

    return null;
}

function populateDestinationSelector(destinations, selectedId) {
    const selector = byId("destination-select");
    selector.replaceChildren();

    for (const destination of destinations) {
        const option = document.createElement("option");
        option.value = destination.id;
        option.textContent = destination.name;
        selector.append(option);
    }

    selector.value = selectedId;
    selector.disabled = false;
}

function clearDestinationLayers() {
    for (const layer of Object.values(state.layers)) {
        removeLayerIfPresent(layer);
    }

    state.layers = {
        regionalImagery: null,
        detailImagery: null,
        cost: null,
        routeCasing: null,
        routeLine: null,
        endpoints: null,
        destinationPins: null,
    };
}

async function loadDestination(destination) {
    const [mission, route] = await Promise.all([
        loadJSON(destination.mission_file),
        loadJSON(destination.route_file),
    ]);

    validateMissionData(mission, route);

    await Promise.all([
        verifyImage(
            imageURL(mission.detail.surface_file),
            mission.detail.width,
            mission.detail.height
        ),
        verifyImage(
            imageURL(mission.detail.cost_file),
            mission.detail.width,
            mission.detail.height
        ),
    ]);

    state.mission = mission;
    state.route = route;
    state.selectedDestination = destination;

    byId("destination-name").textContent = destination.name;
    byId("destination-select").value = destination.id;

    byId("route-status").textContent =
        `${mission.route.cells.toLocaleString()} CELLS · EXPORTED`;

    if (state.map) {
        clearDestinationLayers();
        createMapLayers();
        updateLayerControls();
        fitCurrentView();
    }

    populateRouteMetrics();
    renderElevationProfile();

    console.info(`Loaded destination: ${destination.name}`, {
        routePoints: route.coordinates.length,
        routeMetrics: mission.route,
    });
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

function fitMapToImage(map, bounds, imageWidth, imageHeight) {
    map.invalidateSize({ pan: false });
    map.setMaxBounds(bounds);

    map.fitBounds(bounds, {
        animate: false,
        padding: [MAP_PADDING, MAP_PADDING],
    });

    const baseZoom = map.getZoom();
    const viewport = map.getSize();

    const availableWidth = Math.max(1, viewport.x - MAP_PADDING * 2);
    const availableHeight = Math.max(1, viewport.y - MAP_PADDING * 2);

    const viewportAspect = availableWidth / availableHeight;
    const imageAspect = imageWidth / imageHeight;

    const zoomBoost = Math.max(
        0,
        Math.log2(viewportAspect / imageAspect)
    );

    const targetZoom = Math.min(
        map.getMaxZoom(),
        baseZoom + zoomBoost
    );

    map.setMinZoom(baseZoom - 0.5);

    map.setZoom(targetZoom, {
        animate: false,
    });

    map.invalidateSize({ pan: false });
}

function fitCurrentView() {
    if (!state.map || !state.mission) return;

    const regional = state.activeView === "regional";
    const image = regional
        ? state.mission.overview
        : state.mission.detail;

    const bounds = imageBounds(image.width, image.height);

    fitMapToImage(
        state.map,
        bounds,
        image.width,
        image.height
    );
}

function routeToLatLng(coordinates, imageHeight) {
    return coordinates.map((point) => [
        imageHeight - point.y,
        point.x,
    ]);
}

function imageURL(filename) {
    return new URL(
        filename,
        new URL(DATA_DIRECTORY, window.location.href)
    ).href;
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

    const mars = pixelToMars(
        rasterPixel.x,
        rasterPixel.y,
        transform
    );

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
    const pinsVisible = byId("layer-footprint").checked;
    const costVisible = byId("layer-cost").checked;
    const routeVisible = byId("layer-route").checked;

    const costControl = byId("layer-cost");
    const routeControl = byId("layer-route");
    const opacityControl = byId("cost-opacity");
    const opacityPanel = byId("cost-opacity-control");

    costControl.disabled = regional;
    routeControl.disabled = regional;
    opacityControl.disabled = regional || !costVisible;

    costControl.closest(".layer-option").classList.toggle(
        "is-disabled", regional
    );

    routeControl.closest(".layer-option").classList.toggle(
        "is-disabled", regional
    );

    opacityPanel.classList.toggle(
        "is-disabled",
        regional || !costVisible
    );

    if (regional) {
        if (imageryVisible) {
            addLayerIfMissing(state.layers.regionalImagery);
        } else {
            removeLayerIfPresent(state.layers.regionalImagery);
        }

        if (pinsVisible) {
            addLayerIfMissing(state.layers.destinationPins);
        } else {
            removeLayerIfPresent(state.layers.destinationPins);
        }

        removeLayerIfPresent(state.layers.detailImagery);
        removeLayerIfPresent(state.layers.cost);
        removeLayerIfPresent(state.layers.routeCasing);
        removeLayerIfPresent(state.layers.routeLine);
        removeLayerIfPresent(state.layers.endpoints);
    } else {
        if (imageryVisible) {
            addLayerIfMissing(state.layers.detailImagery);
        } else {
            removeLayerIfPresent(state.layers.detailImagery);
        }

        removeLayerIfPresent(state.layers.regionalImagery);
        removeLayerIfPresent(state.layers.destinationPins);

        if (costVisible) {
            addLayerIfMissing(state.layers.cost);
        } else {
            removeLayerIfPresent(state.layers.cost);
        }

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

    byId("cost-legend").hidden =
        !(state.activeView === "detail" && costVisible);

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

    state.activeView = view;

    updateViewChrome(view);
    updateLayerControls();
    fitCurrentView();

    byId("map-coordinate-readout").textContent =
        "Move cursor over map for coordinates";

    setRouteStatus(
        view === "regional"
            ? "REGIONAL VIEW"
            : "MISSION AREA VIEW"
    );
}

function createDestinationPins() {
    const { mission, scienceSites } = state;
    const overview = mission.overview;

    const scaleX = overview.width / mission.raster.width;
    const scaleY = overview.height / mission.raster.height;

    const markers = [];

    for (const site of scienceSites.destinations) {
        const pixel = site.raster_pixel;

        if (
            !pixel ||
            !Number.isFinite(pixel.x) ||
            !Number.isFinite(pixel.y)
        ) {
            console.warn(`Missing raster pixel for ${site.id}`);
            continue;
        }

        const x = pixel.x * scaleX;
        const y = pixel.y * scaleY;

        const marker = L.marker(
            [overview.height - y, x],
            {
                icon: L.divIcon({
                    className: "science-site-marker",
                    html: `
                        <div class="science-pin">
                            <span class="science-pin-dot"></span>
                            <span class="science-pin-label">
                                ${escapeHTML(site.name)}
                            </span>
                        </div>
                    `,
                    iconSize: [0, 0],
                    iconAnchor: [0, 0],
                }),
                title: site.name,
                keyboard: true,
                alt: site.name,
            }
        );

        marker.on("click", () => openDestinationModal(site));

        markers.push(marker);
    }

    return L.layerGroup(markers);
}

function createMapLayers() {
    const { mission, route } = state;
    const overview = mission.overview;
    const detail = mission.detail;

    const overviewBounds = imageBounds(
        overview.width,
        overview.height
    );

    const detailBounds = imageBounds(
        detail.width,
        detail.height
    );

    state.layers.regionalImagery = L.imageOverlay(
        imageURL(overview.file),
        overviewBounds,
        {
            opacity: 1,
            interactive: false,
            className: "surface-image",
        }
    );

    state.layers.detailImagery = L.imageOverlay(
        imageURL(detail.surface_file),
        detailBounds,
        {
            opacity: 1,
            interactive: false,
            className: "surface-image",
        }
    );

    state.layers.cost = L.imageOverlay(
        imageURL(detail.cost_file),
        detailBounds,
        {
            opacity: DEFAULT_COST_OPACITY,
            interactive: false,
            className: "cost-image",
        }
    );

    const routeLatLngs = routeToLatLng(
        route.coordinates,
        detail.height
    );

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
    }).bindTooltip("START", {
        direction: "top",
        offset: [0, -5],
    });

    const finishMarker = L.circleMarker(finish, {
        radius: 6,
        color: "#071019",
        weight: 2.5,
        fillColor: "#e7a45f",
        fillOpacity: 1,
    }).bindTooltip("DESTINATION", {
        direction: "top",
        offset: [0, -5],
    });

    state.layers.endpoints = L.layerGroup([
        startMarker,
        finishMarker,
    ]);

    state.layers.destinationPins = createDestinationPins();
}

async function initializeMap() {
    if (typeof L === "undefined") {
        throw new Error(
            "Leaflet failed to load. Check vendor/leaflet files."
        );
    }

    const { mission } = state;
    const overview = mission.overview;
    const detail = mission.detail;

    await Promise.all([
        verifyImage(
            imageURL(overview.file),
            overview.width,
            overview.height
        ),
        verifyImage(
            imageURL(detail.surface_file),
            detail.width,
            detail.height
        ),
        verifyImage(
            imageURL(detail.cost_file),
            detail.width,
            detail.height
        ),
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
    state.layers.destinationPins.addTo(map);

    fitCurrentView();

    map.on("mousemove", (event) => {
        updateCoordinateReadout(event.latlng);
    });

    map.on("mouseout", () => {
        byId("map-coordinate-readout").textContent =
            "Move cursor over map for coordinates";
    });

    map.on("click", (event) => {
        if (state.activeView === "regional") {
            const pixel = mapPointToImagePixel(
                event.latlng,
                overview.height
            );

            console.info("Regional image pixel:", pixel);
        } else {
            const pixel = mapPointToImagePixel(
                event.latlng,
                detail.height
            );

            console.info("Mission-area image pixel:", pixel);
        }
    });

    map.whenReady(() => {
        map.invalidateSize();
        fitCurrentView();
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

function openDestinationModal(site) {
    const dialog = byId("destination-modal");
    const content = byId("destination-modal-content");

    if (!dialog || !content) {
        console.error("Destination modal markup is missing.");
        return;
    }

    const coordinates = site.coordinates || {};
    const evidence = Array.isArray(site.evidence) ? site.evidence : [];
    const observations = Array.isArray(site.observations)
        ? site.observations
        : [];

    const evidenceHTML = evidence.length
        ? evidence.map((item) => `
            <article class="modal-evidence">
                <div class="modal-eyebrow">
                    ${escapeHTML(item.evidence_class || item.type || "Evidence")}
                </div>
                <p>${escapeHTML(item.claim)}</p>
                ${item.spatial_relation
                    ? `<p class="modal-secondary">${escapeHTML(item.spatial_relation)}</p>`
                    : ""}
                ${item.confidence
                    ? `<p class="modal-secondary"><strong>Confidence:</strong> ${escapeHTML(item.confidence)}</p>`
                    : ""}
            </article>
        `).join("")
        : `<p class="modal-secondary">No evidence records are available.</p>`;

    const observationsHTML = observations.length
        ? observations.map((item) => {
            const imagePath = item.web_asset
                ? imageURL(item.web_asset)
                : null;

            const sourceURL = safeExternalURL(item.url);

            return `
                <article class="modal-observation">
                    ${imagePath
                        ? `
                            <a class="modal-observation-image"
                               href="${escapeHTML(imagePath)}"
                               target="_blank"
                               rel="noopener noreferrer">
                                <img src="${escapeHTML(imagePath)}"
                                     alt="${escapeHTML(item.asset_id || "Science observation")}"
                                     loading="lazy">
                            </a>
                        `
                        : ""}
                    <div class="modal-observation-info">
                        <div class="modal-eyebrow">
                            ${escapeHTML(item.asset_id || "Observation")}
                            · ${escapeHTML(item.instrument || "Instrument unspecified")}
                        </div>
                        <p>${escapeHTML(item.description)}</p>
                        ${item.relation_to_waypoint
                            ? `<p class="modal-secondary">${escapeHTML(item.relation_to_waypoint)}</p>`
                            : ""}
                        ${sourceURL
                            ? `<a href="${escapeHTML(sourceURL)}" target="_blank" rel="noopener noreferrer">View NASA source ↗</a>`
                            : ""}
                    </div>
                </article>
            `;
        }).join("")
        : `<p class="modal-secondary">No image observations are currently catalogued for this site.</p>`;

    const sourceIDs = Array.isArray(site.sources) ? site.sources : [];
    const sourceRegistry = Array.isArray(state.scienceSites?.source_registry)
        ? state.scienceSites.source_registry
        : [];
    const sourceById = new Map(
        sourceRegistry.map((source) => [source.id, source])
    );

    const sourcesHTML = sourceIDs.length
        ? sourceIDs.map((id) => {
            const source = sourceById.get(id);

            if (!source) {
                return `
                    <article class="modal-source-entry">
                        <span class="modal-source-id">${escapeHTML(id)}</span>
                        <p class="modal-source-missing">Source metadata is not available in the catalogue.</p>
                    </article>
                `;
            }

            const sourceURL = safeExternalURL(source.url);
            const metadata = [source.year, source.publisher]
                .filter(Boolean)
                .join(" · ");

            return `
                <article class="modal-source-entry">
                    <div class="modal-source-heading">
                        <span class="modal-source-id">${escapeHTML(source.id)}</span>
                        ${metadata
                            ? `<span class="modal-source-metadata">${escapeHTML(metadata)}</span>`
                            : ""}
                    </div>
                    <p class="modal-source-title">${escapeHTML(source.title || source.id)}</p>
                    ${source.role
                        ? `<p class="modal-source-role">${escapeHTML(source.role)}</p>`
                        : ""}
                    ${sourceURL
                        ? `<a href="${escapeHTML(sourceURL)}" target="_blank" rel="noopener noreferrer">Open source ↗</a>`
                        : ""}
                </article>
            `;
        }).join("")
        : `<p class="modal-secondary">No source references listed.</p>`;

    content.innerHTML = `
        <div class="modal-site-header">
            <div class="modal-eyebrow">SCIENCE DESTINATION</div>
            <h2>${escapeHTML(site.name)}</h2>
            <div class="modal-geology">${escapeHTML(site.geological_unit || "")}</div>
            <div class="modal-coordinates">
                ${Number.isFinite(coordinates.longitude_deg)
                    ? `${coordinates.longitude_deg.toFixed(3)}° E`
                    : "Longitude unavailable"}
                ·
                ${Number.isFinite(coordinates.latitude_deg)
                    ? `${coordinates.latitude_deg.toFixed(3)}° N`
                    : "Latitude unavailable"}
            </div>
        </div>

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
            <h3>Mission imagery and observations</h3>
            <div class="modal-observations">
                ${observationsHTML}
            </div>
        </section>

        <section class="modal-section">
            <h3>Research sources</h3>
            <div class="modal-source-list">
                ${sourcesHTML}
            </div>
        </section>
    `;

    content.querySelectorAll(".modal-observation-image img").forEach((image) => {
        image.addEventListener("error", () => {
            const imageLink = image.closest(".modal-observation-image");

            if (imageLink) {
                imageLink.hidden = true;

                const note = document.createElement("p");
                note.className = "modal-secondary";
                note.textContent =
                    "The local image asset could not be loaded. Use the source link below.";
                imageLink.insertAdjacentElement("afterend", note);
            }
        }, { once: true });
    });

    const routeButton = byId("modal-route-button");
    routeButton.dataset.destinationId = site.id;

    if (!dialog.open) {
        dialog.showModal();
    }
}

async function routeToDestination(destinationId) {
    const destination = state.destinations.find(
        (entry) => entry.id === destinationId
    );

    if (!destination) {
        throw new Error(`No precomputed route exists for ${destinationId}.`);
    }

    const selector = byId("destination-select");
    const routeButton = byId("modal-route-button");

    selector.disabled = true;
    routeButton.disabled = true;

    setStatus("LOADING PRECOMPUTED ROUTE");

    try {
        await loadDestination(destination);

        switchView("detail");

        byId("destination-modal").close();

        setStatus("OPERATIONAL");

        console.info(
            `Loaded precomputed basecamp route to ${destination.name}`
        );
    } catch (error) {
        console.error("Could not load destination route:", error);

        selector.value = state.selectedDestination?.id || "";

        setRouteStatus(`ROUTE LOAD FAILED: ${error.message}`);
        setStatus("DATA ERROR");
    } finally {
        selector.disabled = false;
        routeButton.disabled = false;
    }
}

function populateRouteMetrics() {
    const route = state.mission.route;

    byId("metric-distance").innerHTML =
        `${route.distance_m.toFixed(1)} <small>m</small>`;

    byId("metric-cost").innerHTML =
        `${route.modeled_cost_m_equivalent.toFixed(1)} <small>m-eq</small>`;

    byId("metric-cells").textContent =
        route.cells.toLocaleString();

    byId("route-status").textContent =
        `${route.cells.toLocaleString()} CELLS · EXPORTED`;

    byId("expanded-cells").textContent =
        route.expanded.toLocaleString();
}

function renderElevationProfile() {
    const container = byId("elevation-profile");

    const profile =
        state.route.elevation_profile ||
        state.mission.route.elevation_profile;

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
        container.textContent =
            "Insufficient valid DEM samples for an elevation profile.";
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

    const area =
        `${padX},${height - padY} ${polyline} ` +
        `${width - padX},${height - padY}`;

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
            <line x1="${padX}" y1="${height - padY}"
                  x2="${width - padX}" y2="${height - padY}"
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
    byId("view-regional").addEventListener(
        "click",
        () => switchView("regional")
    );

    byId("view-detail").addEventListener(
        "click",
        () => switchView("detail")
    );

    byId("destination-select").addEventListener(
        "change",
        async (event) => {
            const destination = state.destinations.find(
                (entry) => entry.id === event.target.value
            );

            if (!destination) return;

            const selector = byId("destination-select");
            selector.disabled = true;

            setStatus("LOADING DESTINATION");

            try {
                await loadDestination(destination);
                setStatus("OPERATIONAL");
            } catch (error) {
                console.error(
                    "Could not load selected destination:",
                    error
                );

                selector.value =
                    state.selectedDestination?.id || "";

                setRouteStatus(
                    `DESTINATION LOAD FAILED: ${error.message}`
                );

                setStatus("DATA ERROR");
            } finally {
                selector.disabled = false;
            }
        }
    );

    for (const id of [
        "layer-imagery",
        "layer-footprint",
        "layer-cost",
        "layer-route",
    ]) {
        byId(id).addEventListener(
            "change",
            updateLayerControls
        );
    }

    byId("cost-opacity").addEventListener(
        "input",
        (event) => {
            const opacity = Number(event.target.value) / 100;

            if (state.layers.cost) {
                state.layers.cost.setOpacity(opacity);
            }

            byId("cost-opacity-value").value =
                `${event.target.value}%`;

            byId("cost-opacity-value").textContent =
                `${event.target.value}%`;
        }
    );

    byId("map-hint").addEventListener(
        "click",
        () => switchView("detail")
    );

    byId("modal-close").addEventListener(
        "click",
        () => byId("destination-modal").close()
    );

    byId("modal-route-button").addEventListener(
        "click",
        async () => {
            const destinationId =
                byId("modal-route-button").dataset.destinationId;

            await routeToDestination(destinationId);
        }
    );

    byId("destination-modal").addEventListener(
        "click",
        (event) => {
            if (event.target === byId("destination-modal")) {
                byId("destination-modal").close();
            }
        }
    );
}

async function initializeARES() {
    try {
        setStatus("LOADING DATA");

        const [destinationIndex, scienceCatalogue] = await Promise.all([
            loadJSON("destinations.json"),
            loadJSON("science-sites.json"),
        ]);

        if (
            !Array.isArray(destinationIndex.destinations) ||
            destinationIndex.destinations.length === 0
        ) {
            throw new Error(
                "Destination index is missing or contains no destinations."
            );
        }

        if (!Array.isArray(scienceCatalogue.destinations)) {
            throw new Error(
                "Science catalogue is missing its destinations array."
            );
        }

        state.destinations = destinationIndex.destinations;
        state.scienceSites = scienceCatalogue;

        const missingRoute = scienceCatalogue.destinations.find(
            (site) => !state.destinations.some(
                (destination) => destination.id === site.id
            )
        );

        if (missingRoute) {
            throw new Error(
                `No precomputed route bundle found for ${missingRoute.id}.`
            );
        }

        const initialDestination = state.destinations[0];

        populateDestinationSelector(
            state.destinations,
            initialDestination.id
        );

        await loadDestination(initialDestination);
        await initializeMap();

        connectControls();
        updateViewChrome("regional");
        updateLayerControls();

        setStatus("OPERATIONAL");

        window.ARES = {
            map: state.map,
            mission: state.mission,
            route: state.route,
            destinations: state.destinations,
            scienceSites: state.scienceSites,
            selectedDestination: state.selectedDestination,
            layers: state.layers,
            switchView,
            loadDestination,
            openDestinationModal,
            routeToDestination,
            updateLayerControls,
            fitCurrentView,
        };

        console.info("ARES initialized.", {
            destinationCount: state.destinations.length,
            scienceSiteCount: state.scienceSites.destinations.length,
            selectedDestination: state.selectedDestination,
            overview: state.mission.overview,
            detail: state.mission.detail,
            routePoints: state.route.coordinates.length,
            routeMetrics: state.mission.route,
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

        mapElement.querySelector(
            ".map-error-message"
        ).textContent = error.message;

        setStatus("INITIALIZATION ERROR");
    }
}

document.addEventListener(
    "DOMContentLoaded",
    initializeARES
);
