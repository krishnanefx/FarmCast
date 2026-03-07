import {
    prettyAgeHours,
    prettySource,
    prettyUpdatedAt,
    toCelsius,
    toHpa,
    windDirectionDegrees,
    windMagnitude,
} from "./formatters.js";
import { STATUS } from "./state.js";

const UK_REGION_COORDS = {
    "East Midlands": [52.8, -1.2],
    Yorkshire: [53.9, -1.5],
    "East Anglia": [52.4, 0.9],
    "Southwest England": [50.7, -3.5],
    Wales: [52.1, -3.8],
    Scotland: [56.5, -4.0],
    "Northern Ireland": [54.6, -6.7],
};

const CONTROL_GIFS = {
    tractor: {
        success: "/static/assets/tractor_success.gif",
        fail: "/static/assets/tractor_fail.gif",
    },
    drone: {
        success: "/static/assets/drone_success.gif",
        fail: "/static/assets/drone_fail.gif",
    },
    livestock: {
        success: "/static/assets/cow_success.gif",
        fail: "/static/assets/cow_fail.gif",
    },
};

function clamp(value, min = 0, max = 1) {
    return Math.min(max, Math.max(min, value));
}

function setText(node, value) {
    if (!node) return;
    node.textContent = value;
}

function show(node, visible) {
    if (!node) return;
    node.classList.toggle("hidden", !visible);
}

function sanitizeNumber(value, fallback = 0) {
    return Number.isFinite(value) ? value : fallback;
}

function deriveSurface(data) {
    const temperatureC = sanitizeNumber(toCelsius(data.metrics.temperature_2m));
    const pressureHpa = sanitizeNumber(toHpa(data.metrics.mean_sea_level_pressure));
    const windMs = sanitizeNumber(
        windMagnitude(data.metrics.u_wind_10m, data.metrics.v_wind_10m)
    );
    const windKmh = windMs * 3.6;
    const windDirectionDeg = sanitizeNumber(
        windDirectionDegrees(data.metrics.u_wind_10m, data.metrics.v_wind_10m)
    );
    const pressureFactor = clamp((1015 - pressureHpa) / 35);
    const windFactor = clamp(windMs / 24) * 0.22;
    const precipitationProbability = clamp(pressureFactor + windFactor);
    const precipitationMm = precipitationProbability * 3.0;
    const frostRisk = temperatureC < 2;

    return {
        temperatureC,
        pressureHpa,
        windMs,
        windKmh,
        windDirectionDeg,
        precipitationProbability,
        precipitationMm,
        frostRisk,
    };
}

function classifyCondition(surface) {
    if (surface.frostRisk) return "Frost risk";
    if (surface.precipitationProbability > 0.65) return "Heavy rain likely";
    if (surface.precipitationProbability > 0.35) return "Showers possible";
    if (surface.windKmh > 32) return "Windy";
    return "Clear conditions";
}

function evaluateDecisions(surface) {
    const decisions = [];

    if (surface.windKmh > 22 || surface.precipitationProbability > 0.5) {
        decisions.push({
            key: "spraying",
            icon: "🌿",
            activity: "Pesticide Spraying",
            verdict: "avoid",
            reason: "Spray drift/wash-off risk is high.",
            signal: `Wind ${surface.windKmh.toFixed(0)} km/h · Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`,
        });
    } else if (surface.windKmh > 14 || surface.precipitationProbability > 0.3) {
        decisions.push({
            key: "spraying",
            icon: "🌿",
            activity: "Pesticide Spraying",
            verdict: "caution",
            reason: "Conditions are workable but not optimal.",
            signal: `Wind ${surface.windKmh.toFixed(0)} km/h · Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`,
        });
    } else {
        decisions.push({
            key: "spraying",
            icon: "🌿",
            activity: "Pesticide Spraying",
            verdict: "good",
            reason: "Wind and rain risk are within safe range.",
            signal: `Wind ${surface.windKmh.toFixed(0)} km/h · Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`,
        });
    }

    if (surface.precipitationMm > 2.2) {
        decisions.push({
            key: "fieldwork",
            icon: "🚜",
            activity: "Field Work / Tilling",
            verdict: "avoid",
            reason: "Soil compaction and traction risk are high.",
            signal: `Expected rain ${surface.precipitationMm.toFixed(1)} mm`,
        });
    } else if (surface.precipitationMm > 1.0) {
        decisions.push({
            key: "fieldwork",
            icon: "🚜",
            activity: "Field Work / Tilling",
            verdict: "caution",
            reason: "Check field moisture before committing machinery.",
            signal: `Expected rain ${surface.precipitationMm.toFixed(1)} mm`,
        });
    } else {
        decisions.push({
            key: "fieldwork",
            icon: "🚜",
            activity: "Field Work / Tilling",
            verdict: "good",
            reason: "Ground conditions should support normal operations.",
            signal: `Expected rain ${surface.precipitationMm.toFixed(1)} mm`,
        });
    }

    if (surface.frostRisk) {
        decisions.push({
            key: "livestock",
            icon: "🐄",
            activity: "Livestock Outdoors",
            verdict: "caution",
            reason: "Protect vulnerable stock from overnight frost.",
            signal: `Temp ${surface.temperatureC.toFixed(1)}°C`,
        });
    } else {
        decisions.push({
            key: "livestock",
            icon: "🐄",
            activity: "Livestock Outdoors",
            verdict: "good",
            reason: "No frost signal and manageable wind.",
            signal: `Temp ${surface.temperatureC.toFixed(1)}°C`,
        });
    }

    if (surface.frostRisk || surface.temperatureC < 4) {
        decisions.push({
            key: "planting",
            icon: "🌱",
            activity: "Planting",
            verdict: "caution",
            reason: "Low temperatures may reduce early growth stability.",
            signal: `Temp ${surface.temperatureC.toFixed(1)}°C`,
        });
    } else if (surface.precipitationProbability > 0.7) {
        decisions.push({
            key: "planting",
            icon: "🌱",
            activity: "Planting",
            verdict: "avoid",
            reason: "Rain intensity risk is high for immediate planting.",
            signal: `Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`,
        });
    } else {
        decisions.push({
            key: "planting",
            icon: "🌱",
            activity: "Planting",
            verdict: "good",
            reason: "Temperature and rain profile are suitable.",
            signal: `Temp ${surface.temperatureC.toFixed(1)}°C`,
        });
    }

    return decisions;
}

function decisionBadgeLabel(verdict) {
    if (verdict === "good") return "✓ Good";
    if (verdict === "caution") return "⚠ Caution";
    return "✕ Avoid";
}

function decisionRank(verdict) {
    if (verdict === "avoid") return 3;
    if (verdict === "caution") return 2;
    return 1;
}

function summarizeForSpeech(data, surface, decisions) {
    const location = data.location || "the selected location";
    const highest = decisions.reduce((current, next) =>
        decisionRank(next.verdict) > decisionRank(current.verdict) ? next : current
    );

    return [
        `FarmCast update for ${location}.`,
        `Temperature ${surface.temperatureC.toFixed(1)} degrees Celsius.`,
        `Wind ${surface.windKmh.toFixed(0)} kilometres per hour.`,
        `Rain probability ${(surface.precipitationProbability * 100).toFixed(0)} percent.`,
        `Frost risk is ${surface.frostRisk ? "present" : "not present"}.`,
        `Highest-risk activity is ${highest.activity}: ${decisionBadgeLabel(highest.verdict)}.`,
    ].join(" ");
}

export function createRenderer() {
    const refs = {
        form: document.getElementById("weather-form"),
        locationInput: document.getElementById("location"),
        modelSelect: document.getElementById("model-select"),
        submitBtn: document.getElementById("submit-btn"),
        refreshBtn: document.getElementById("refresh-btn"),
        btnText: document.querySelector("#submit-btn .btn-text"),
        loader: document.querySelector("#submit-btn .loader"),
        statusBanner: document.getElementById("status-banner"),
        srStatus: document.getElementById("screen-reader-status"),
        loadingContainer: document.getElementById("loading-container"),
        resultsContainer: document.getElementById("results-container"),
        errorContainer: document.getElementById("error-container"),
        errorMessage: document.getElementById("error-message"),
        errorHelp: document.getElementById("error-help"),
        resultLocation: document.getElementById("result-location"),
        heroTemp: document.getElementById("hero-temp"),
        heroCondition: document.getElementById("hero-condition"),
        heroSummary: document.getElementById("hero-summary"),
        resultCoords: document.getElementById("result-coords"),
        resultSource: document.getElementById("result-source"),
        resultModel: document.getElementById("result-model"),
        resultAge: document.getElementById("result-age"),
        resultUpdated: document.getElementById("result-updated"),
        metricTemp: document.getElementById("metric-temp"),
        metricWind: document.getElementById("metric-wind"),
        metricPressure: document.getElementById("metric-pressure"),
        metricDirection: document.getElementById("metric-direction"),
        metricRain: document.getElementById("metric-rain"),
        diagnosticsList: document.getElementById("diagnostics-list"),
        decisionGrid: document.getElementById("decision-grid"),
        ttsBtn: document.getElementById("tts-btn"),
        textScale: document.getElementById("text-scale"),
        highContrastToggle: document.getElementById("high-contrast-toggle"),
        reduceMotionToggle: document.getElementById("reduce-motion-toggle"),
        tractorBtn: document.getElementById("control-tractor-btn"),
        droneBtn: document.getElementById("control-drone-btn"),
        livestockBtn: document.getElementById("control-livestock-btn"),
        tractorStatus: document.getElementById("control-tractor-status"),
        droneStatus: document.getElementById("control-drone-status"),
        livestockStatus: document.getElementById("control-livestock-status"),
        tractorGif: document.getElementById("control-tractor-gif"),
        droneGif: document.getElementById("control-drone-gif"),
        livestockGif: document.getElementById("control-livestock-gif"),
        mapContainer: document.getElementById("forecast-map"),
        mapBaseInputs: Array.from(document.querySelectorAll('input[name="map-base"]')),
        layerActive: document.getElementById("map-layer-active"),
        layerRisk: document.getElementById("map-layer-risk"),
        layerWind: document.getElementById("map-layer-wind"),
        layerRain: document.getElementById("map-layer-rain"),
        layerGrid: document.getElementById("map-layer-grid"),
        layerWorld: document.getElementById("map-layer-world"),
    };

    const latest = {
        data: null,
        surface: null,
        decisions: [],
    };

    const controls = {
        tractor: { text: "Status: idle", gif: null, alt: "" },
        drone: { text: "Status: idle", gif: null, alt: "" },
        livestock: { text: "Status: idle", gif: null, alt: "" },
    };

    const mapState = {
        map: null,
        baseLayers: {},
        activeBase: "dark",
        overlays: {},
        worldLoaded: false,
    };

    function setStatusMessage(message) {
        setText(refs.statusBanner, message);
        setText(refs.srStatus, message);
    }

    function applyA11ySettings() {
        const textScale = Number.parseInt(refs.textScale.value, 10) || 16;
        document.documentElement.style.fontSize = `${textScale}px`;
        document.body.dataset.highContrast = refs.highContrastToggle.checked ? "true" : "false";
        document.body.dataset.reduceMotion = refs.reduceMotionToggle.checked ? "true" : "false";
    }

    function setGif(imageNode, source, altText) {
        if (!source) {
            imageNode.src = "";
            imageNode.alt = "";
            show(imageNode, false);
            return;
        }
        imageNode.onerror = () => {
            imageNode.src = "";
            imageNode.alt = "";
            show(imageNode, false);
        };
        imageNode.src = source;
        imageNode.alt = altText;
        show(imageNode, true);
    }

    function renderControlStatuses() {
        setText(refs.tractorStatus, controls.tractor.text);
        setText(refs.droneStatus, controls.drone.text);
        setText(refs.livestockStatus, controls.livestock.text);
        setGif(refs.tractorGif, controls.tractor.gif, controls.tractor.alt);
        setGif(refs.droneGif, controls.drone.gif, controls.drone.alt);
        setGif(refs.livestockGif, controls.livestock.gif, controls.livestock.alt);
    }

    function resetControlStatuses() {
        controls.tractor = { text: "Status: idle", gif: null, alt: "" };
        controls.drone = { text: "Status: idle", gif: null, alt: "" };
        controls.livestock = { text: "Status: idle", gif: null, alt: "" };
        renderControlStatuses();
    }

    function renderDecisionCards(decisions) {
        refs.decisionGrid.innerHTML = decisions
            .map((decision) => {
                const badge = decisionBadgeLabel(decision.verdict);
                return `
                    <article class="decision-card" aria-label="${decision.activity} ${badge}">
                        <p class="decision-title">${decision.icon} ${decision.activity}</p>
                        <p class="decision-reason">${decision.reason}</p>
                        <p class="decision-signal">${decision.signal}</p>
                        <span class="decision-badge ${decision.verdict}" role="status">${badge}</span>
                    </article>
                `;
            })
            .join("");
    }

    function renderDiagnostics(data, surface) {
        const diagnostics = [
            `Model path: ${data.model?.file || "unknown"}`,
            `Forecast horizon: ${data.forecast_hours ?? data.model?.hours ?? "--"}h`,
            `Input source: ${prettySource(data.source)}`,
            `Source age: ${prettyAgeHours(data.source)}`,
            `Derived rain probability: ${(surface.precipitationProbability * 100).toFixed(0)}%`,
            `Derived frost risk: ${surface.frostRisk ? "yes" : "no"}`,
        ];

        refs.diagnosticsList.innerHTML = "";
        diagnostics.forEach((line) => {
            const li = document.createElement("li");
            li.textContent = line;
            refs.diagnosticsList.appendChild(li);
        });
    }

    function buildGridLayer(layer) {
        for (let lat = -60; lat <= 60; lat += 30) {
            window.L.polyline(
                [
                    [lat, -180],
                    [lat, 180],
                ],
                { color: "#8bc7e6", weight: 1, opacity: 0.28, interactive: false }
            ).addTo(layer);
        }
        for (let lon = -150; lon <= 180; lon += 30) {
            window.L.polyline(
                [
                    [-80, lon],
                    [80, lon],
                ],
                { color: "#8bc7e6", weight: 1, opacity: 0.24, interactive: false }
            ).addTo(layer);
        }
    }

    function ensureMap() {
        if (!refs.mapContainer || !window.L || mapState.map) return;

        mapState.map = window.L.map(refs.mapContainer, {
            center: [54.5, -3.0],
            zoom: 5,
            minZoom: 2,
            worldCopyJump: true,
            zoomControl: true,
        });

        mapState.baseLayers.dark = window.L.tileLayer(
            "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            { attribution: "&copy; OpenStreetMap &copy; CARTO" }
        );
        mapState.baseLayers.light = window.L.tileLayer(
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            { attribution: "&copy; OpenStreetMap &copy; CARTO" }
        );
        mapState.baseLayers.satellite = window.L.tileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            { attribution: "Tiles &copy; Esri" }
        );
        mapState.baseLayers.dark.addTo(mapState.map);

        mapState.overlays.active = window.L.layerGroup().addTo(mapState.map);
        mapState.overlays.risk = window.L.layerGroup().addTo(mapState.map);
        mapState.overlays.wind = window.L.layerGroup().addTo(mapState.map);
        mapState.overlays.rain = window.L.layerGroup().addTo(mapState.map);
        mapState.overlays.grid = window.L.layerGroup().addTo(mapState.map);
        mapState.overlays.world = window.L.layerGroup().addTo(mapState.map);
        buildGridLayer(mapState.overlays.grid);

        loadWorldBoundaries();
        applyMapVisibility();
    }

    function applyMapVisibility() {
        if (!mapState.map) return;

        const selectedBase = refs.mapBaseInputs.find((input) => input.checked)?.value || "dark";
        if (selectedBase !== mapState.activeBase) {
            mapState.map.removeLayer(mapState.baseLayers[mapState.activeBase]);
            mapState.baseLayers[selectedBase].addTo(mapState.map);
            mapState.activeBase = selectedBase;
        }

        const layerToggles = [
            { checkbox: refs.layerActive, layer: mapState.overlays.active },
            { checkbox: refs.layerRisk, layer: mapState.overlays.risk },
            { checkbox: refs.layerWind, layer: mapState.overlays.wind },
            { checkbox: refs.layerRain, layer: mapState.overlays.rain },
            { checkbox: refs.layerGrid, layer: mapState.overlays.grid },
            { checkbox: refs.layerWorld, layer: mapState.overlays.world },
        ];

        layerToggles.forEach(({ checkbox, layer }) => {
            if (!layer) return;
            const enabled = Boolean(checkbox?.checked);
            const hasLayer = mapState.map.hasLayer(layer);
            if (enabled && !hasLayer) mapState.map.addLayer(layer);
            if (!enabled && hasLayer) mapState.map.removeLayer(layer);
        });
    }

    async function loadWorldBoundaries() {
        if (mapState.worldLoaded || !window.topojson) return;
        try {
            const response = await fetch("/static/data/countries-110m.json");
            if (!response.ok) return;
            const topology = await response.json();
            const objectKey = Object.keys(topology.objects || {})[0];
            if (!objectKey) return;
            const geojson = window.topojson.feature(topology, topology.objects[objectKey]);
            mapState.overlays.world.clearLayers();
            window.L.geoJSON(geojson, {
                style: () => ({
                    color: "#8bc7e6",
                    weight: 0.7,
                    opacity: 0.32,
                    fillOpacity: 0,
                }),
                interactive: false,
            }).addTo(mapState.overlays.world);
            mapState.worldLoaded = true;
            applyMapVisibility();
        } catch {
            mapState.worldLoaded = false;
        }
    }

    function renderMap(data, regionalRows) {
        ensureMap();
        if (!mapState.map) return;

        mapState.overlays.active.clearLayers();
        mapState.overlays.risk.clearLayers();
        mapState.overlays.wind.clearLayers();
        mapState.overlays.rain.clearLayers();

        const lat = data.coordinates?.lat;
        const lon = data.coordinates?.lon;
        const surface = latest.surface;

        if (Number.isFinite(lat) && Number.isFinite(lon)) {
            window.L.marker([lat, lon])
                .bindPopup(
                    `<strong>${data.location || "Active location"}</strong><br>` +
                        `Temp ${surface.temperatureC.toFixed(1)}°C<br>` +
                        `Wind ${surface.windKmh.toFixed(0)} km/h<br>` +
                        `Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`
                )
                .addTo(mapState.overlays.active);

            window.L.circle([lat, lon], {
                radius: 130000 + Math.round(surface.precipitationProbability * 170000),
                color: "#7dd3fc",
                weight: 2,
                fillColor: "#7dd3fc",
                fillOpacity: 0.16,
            }).addTo(mapState.overlays.active);
        }

        regionalRows.forEach((row) => {
            if (!row?.data) return;
            const coords = UK_REGION_COORDS[row.region];
            if (!coords) return;

            const regionalSurface = deriveSurface(row.data);
            const regionalDecisions = evaluateDecisions(regionalSurface);
            const worst = regionalDecisions.reduce((current, next) =>
                decisionRank(next.verdict) > decisionRank(current.verdict) ? next : current
            );

            const riskColor =
                worst.verdict === "avoid"
                    ? "#ff8f8f"
                    : worst.verdict === "caution"
                      ? "#ffe08a"
                      : "#9ef6b3";

            window.L.circleMarker(coords, {
                radius: 8,
                color: riskColor,
                fillColor: riskColor,
                fillOpacity: 0.85,
                weight: 1,
            })
                .bindTooltip(`${row.region}: ${decisionBadgeLabel(worst.verdict)}`)
                .addTo(mapState.overlays.risk);

            window.L.circleMarker(coords, {
                radius: clamp(4 + regionalSurface.windKmh / 8, 5, 16),
                color: "#7dd3fc",
                fillColor: "#38bdf8",
                fillOpacity: 0.45,
                weight: 1.2,
            })
                .bindTooltip(`${row.region}: wind ${regionalSurface.windKmh.toFixed(0)} km/h`)
                .addTo(mapState.overlays.wind);

            const rainColor =
                regionalSurface.precipitationProbability >= 0.66
                    ? "#ff8f8f"
                    : regionalSurface.precipitationProbability >= 0.33
                      ? "#ffe08a"
                      : "#9ef6b3";

            window.L.circleMarker(coords, {
                radius: clamp(4 + regionalSurface.precipitationProbability * 14, 4, 15),
                color: rainColor,
                fillColor: rainColor,
                fillOpacity: 0.46,
                weight: 1.2,
            })
                .bindTooltip(
                    `${row.region}: rain ${(regionalSurface.precipitationProbability * 100).toFixed(0)}%`
                )
                .addTo(mapState.overlays.rain);
        });

        applyMapVisibility();
        setTimeout(() => mapState.map.invalidateSize(), 60);
    }

    function enableActionButtons(enabled) {
        refs.tractorBtn.disabled = !enabled;
        refs.droneBtn.disabled = !enabled;
        refs.livestockBtn.disabled = !enabled;
        refs.ttsBtn.disabled = !enabled;
        refs.refreshBtn.disabled = !enabled;
    }

    function onTractorAction() {
        if (!latest.surface) return;
        if (latest.surface.precipitationMm >= 2.0) {
            controls.tractor = {
                text: `Status: ✕ Avoid, soil too wet (${latest.surface.precipitationMm.toFixed(1)} mm).`,
                gif: CONTROL_GIFS.tractor.fail,
                alt: "Tractor blocked due to wet soil conditions.",
            };
        } else {
            controls.tractor = {
                text: "Status: ✓ Good, tractor dispatched.",
                gif: CONTROL_GIFS.tractor.success,
                alt: "Tractor operating successfully in suitable conditions.",
            };
        }
        renderControlStatuses();
    }

    function onDroneAction() {
        if (!latest.surface) return;
        if (latest.surface.windKmh >= 15 || latest.surface.precipitationMm > 0) {
            controls.drone = {
                text: "Status: ⚠ Caution, drone grounded due to drift/wash-off risk.",
                gif: CONTROL_GIFS.drone.fail,
                alt: "Drone action blocked by weather risk.",
            };
        } else {
            controls.drone = {
                text: "Status: ✓ Good, drone deployed.",
                gif: CONTROL_GIFS.drone.success,
                alt: "Drone action completed in suitable conditions.",
            };
        }
        renderControlStatuses();
    }

    function onLivestockAction() {
        if (!latest.surface) return;
        if (latest.surface.frostRisk) {
            controls.livestock = {
                text: "Status: ⚠ Caution, frost risk detected.",
                gif: CONTROL_GIFS.livestock.fail,
                alt: "Livestock warning due to frost risk.",
            };
        } else {
            controls.livestock = {
                text: "Status: ✓ Good, pasture conditions safe.",
                gif: CONTROL_GIFS.livestock.success,
                alt: "Livestock conditions are safe for outdoor grazing.",
            };
        }
        renderControlStatuses();
    }

    function onSpeakForecast() {
        if (!latest.data || !latest.surface) return;
        if (!window.speechSynthesis) {
            setStatusMessage("Browser speech synthesis is not available.");
            return;
        }
        const script = summarizeForSpeech(latest.data, latest.surface, latest.decisions);
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(script);
        utterance.lang = "en-GB";
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
        setStatusMessage("Reading forecast aloud.");
    }

    function bindControls() {
        if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            refs.reduceMotionToggle.checked = true;
        }
        applyA11ySettings();

        refs.textScale.addEventListener("input", applyA11ySettings);
        refs.highContrastToggle.addEventListener("change", applyA11ySettings);
        refs.reduceMotionToggle.addEventListener("change", applyA11ySettings);

        refs.tractorBtn.addEventListener("click", onTractorAction);
        refs.droneBtn.addEventListener("click", onDroneAction);
        refs.livestockBtn.addEventListener("click", onLivestockAction);
        refs.ttsBtn.addEventListener("click", onSpeakForecast);

        refs.mapBaseInputs.forEach((node) =>
            node.addEventListener("change", applyMapVisibility)
        );
        refs.layerActive.addEventListener("change", applyMapVisibility);
        refs.layerRisk.addEventListener("change", applyMapVisibility);
        refs.layerWind.addEventListener("change", applyMapVisibility);
        refs.layerRain.addEventListener("change", applyMapVisibility);
        refs.layerGrid.addEventListener("change", applyMapVisibility);
        refs.layerWorld.addEventListener("change", applyMapVisibility);
    }

    function renderLoadingState(state) {
        refs.submitBtn.disabled = true;
        refs.form.setAttribute("aria-busy", "true");
        refs.btnText.classList.add("hidden");
        refs.loader.classList.remove("hidden");
        show(refs.loadingContainer, true);
        show(refs.errorContainer, false);
        show(refs.resultsContainer, false);
        refs.refreshBtn.disabled = true;
        const label = state.lastAction === "refresh" ? "Refreshing forecast..." : "Running forecast query...";
        setStatusMessage(label);
    }

    function renderIdleState(state) {
        refs.submitBtn.disabled = false;
        refs.form.setAttribute("aria-busy", "false");
        refs.btnText.classList.remove("hidden");
        refs.loader.classList.add("hidden");
        show(refs.loadingContainer, false);
        refs.refreshBtn.disabled = !state.lastPayload;
        if (!latest.data) {
            setStatusMessage("Ready. Enter a location to run a forecast.");
        }
    }

    function renderErrorState(state) {
        renderIdleState(state);
        show(refs.resultsContainer, false);
        show(refs.errorContainer, true);
        setText(refs.errorMessage, state.error?.message || "Forecast request failed.");
        if (refs.errorHelp) {
            refs.errorHelp.innerHTML = `<p>${state.error?.help || "Check model files, input fields, and location spelling, then retry."}</p>`;
        }
        setStatusMessage("Forecast failed. Review the error details.");
        enableActionButtons(false);
        resetControlStatuses();
    }

    function renderSuccessState(state) {
        renderIdleState(state);
        show(refs.errorContainer, false);
        show(refs.resultsContainer, true);

        latest.data = state.data;
        latest.surface = deriveSurface(state.data);
        latest.decisions = evaluateDecisions(latest.surface);

        const surface = latest.surface;
        const condition = classifyCondition(surface);
        const actionText = state.lastAction === "refresh" ? "refresh complete" : "forecast loaded";

        setText(refs.resultLocation, state.data.location || "Unknown location");
        setText(refs.heroTemp, `${surface.temperatureC.toFixed(1)} °C`);
        setText(refs.heroCondition, condition);
        setText(
            refs.heroSummary,
            `Wind ${surface.windKmh.toFixed(0)} km/h, Rain ${(surface.precipitationProbability * 100).toFixed(0)}%`
        );
        setText(
            refs.resultCoords,
            `Lat: ${state.data.coordinates.lat.toFixed(4)}, Lon: ${state.data.coordinates.lon.toFixed(4)} | Grid: [${state.data.grid_idx.lat_idx}, ${state.data.grid_idx.lon_idx}]`
        );

        setText(refs.resultSource, prettySource(state.data.source));
        setText(refs.resultModel, `${state.data.model?.hours || "--"}h`);
        setText(refs.resultAge, prettyAgeHours(state.data.source));
        setText(refs.resultUpdated, prettyUpdatedAt(state.data.source, state.data.forecast_time));

        setText(refs.metricTemp, `${surface.temperatureC.toFixed(1)} °C`);
        setText(refs.metricWind, `${surface.windMs.toFixed(1)} m/s`);
        setText(refs.metricPressure, `${Math.round(surface.pressureHpa)} hPa`);
        setText(refs.metricDirection, `${surface.windDirectionDeg.toFixed(0)}°`);
        setText(refs.metricRain, `Rain probability ${(surface.precipitationProbability * 100).toFixed(0)}%`);

        renderDecisionCards(latest.decisions);
        renderDiagnostics(state.data, surface);
        renderMap(state.data, state.data.regional || []);

        enableActionButtons(true);
        resetControlStatuses();

        const timeLabel = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        setStatusMessage(`Forecast ${actionText} for ${state.data.location} at ${timeLabel}.`);
        refs.resultsContainer.focus();
    }

    function render(state) {
        if (state.status === STATUS.LOADING) {
            renderLoadingState(state);
            return;
        }
        if (state.status === STATUS.ERROR) {
            renderErrorState(state);
            return;
        }
        if (state.status === STATUS.SUCCESS && state.data) {
            renderSuccessState(state);
            return;
        }
        renderIdleState(state);
    }

    function getRequestPayload() {
        return {
            location: refs.locationInput.value.trim(),
            model: refs.modelSelect.value,
        };
    }

    function getRefreshPayload(lastPayload) {
        return {
            location: refs.locationInput.value.trim() || lastPayload?.location || "",
            model: refs.modelSelect.value || lastPayload?.model || "1",
        };
    }

    bindControls();
    resetControlStatuses();
    enableActionButtons(false);

    return {
        refs,
        render,
        getRequestPayload,
        getRefreshPayload,
    };
}
