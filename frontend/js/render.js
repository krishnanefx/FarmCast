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

function setText(node, value) {
    if (!node) return;
    node.textContent = value;
}

function show(node, visible) {
    if (!node) return;
    node.classList.toggle("hidden", !visible);
}

export function createRenderer() {
    const refs = {
        form: document.getElementById("weather-form"),
        locationInput: document.getElementById("location"),
        modelSelect: document.getElementById("model-select"),
        submitBtn: document.getElementById("submit-btn"),
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
        diagnosticsList: document.getElementById("diagnostics-list"),
    };

    function setStatusMessage(message) {
        setText(refs.statusBanner, message);
        setText(refs.srStatus, message);
    }

    function renderLoadingState() {
        refs.submitBtn.disabled = true;
        refs.form.setAttribute("aria-busy", "true");
        refs.btnText.classList.add("hidden");
        refs.loader.classList.remove("hidden");
        show(refs.loadingContainer, true);
        show(refs.errorContainer, false);
        show(refs.resultsContainer, false);
        setStatusMessage("Running forecast query...");
    }

    function renderIdleState() {
        refs.submitBtn.disabled = false;
        refs.form.setAttribute("aria-busy", "false");
        refs.btnText.classList.remove("hidden");
        refs.loader.classList.add("hidden");
        show(refs.loadingContainer, false);
        setStatusMessage("Ready. Enter a location to run a forecast.");
    }

    function renderErrorState(error) {
        renderIdleState();
        show(refs.resultsContainer, false);
        show(refs.errorContainer, true);
        setText(refs.errorMessage, error?.message || "Forecast request failed.");
        if (refs.errorHelp) {
            refs.errorHelp.innerHTML = `<p>${error?.help || "Check model files, input fields, and location spelling, then retry."}</p>`;
        }
        setStatusMessage("Forecast failed. Review the error details.");
    }

    function renderSuccessState(data) {
        renderIdleState();
        show(refs.errorContainer, false);
        show(refs.resultsContainer, true);

        const tempC = toCelsius(data.metrics.temperature_2m);
        const pressureHpa = toHpa(data.metrics.mean_sea_level_pressure);
        const windMs = windMagnitude(data.metrics.u_wind_10m, data.metrics.v_wind_10m);
        const directionDeg = windDirectionDegrees(data.metrics.u_wind_10m, data.metrics.v_wind_10m);

        setText(refs.resultLocation, data.location || "Unknown location");
        setText(refs.heroTemp, `${tempC.toFixed(1)} °C`);
        setText(refs.heroSummary, `Wind ${windMs.toFixed(1)} m/s, Pressure ${Math.round(pressureHpa)} hPa`);
        setText(
            refs.resultCoords,
            `Lat: ${data.coordinates.lat.toFixed(4)}, Lon: ${data.coordinates.lon.toFixed(4)} | Grid: [${data.grid_idx.lat_idx}, ${data.grid_idx.lon_idx}]`
        );

        setText(refs.resultSource, prettySource(data.source));
        setText(refs.resultModel, `${data.model?.hours || "--"}h`);
        setText(refs.resultAge, prettyAgeHours(data.source));
        setText(refs.resultUpdated, prettyUpdatedAt(data.source, data.forecast_time));

        setText(refs.metricTemp, `${tempC.toFixed(1)} °C`);
        setText(refs.metricWind, `${windMs.toFixed(1)} m/s`);
        setText(refs.metricPressure, `${Math.round(pressureHpa)} hPa`);
        setText(refs.metricDirection, `${directionDeg.toFixed(0)}°`);

        const diagnostics = [
            `Model path: ${data.model?.file || "unknown"}`,
            `Forecast horizon: ${data.forecast_hours ?? data.model?.hours ?? "--"}h`,
            `Input source: ${prettySource(data.source)}`,
            `Source age: ${prettyAgeHours(data.source)}`,
        ];
        refs.diagnosticsList.innerHTML = diagnostics.map((item) => `<li>${item}</li>`).join("");
        refs.resultsContainer.focus();
        setStatusMessage(`Forecast updated for ${data.location}.`);
    }

    function render(state) {
        if (state.status === STATUS.LOADING) {
            renderLoadingState();
            return;
        }
        if (state.status === STATUS.ERROR) {
            renderErrorState(state.error);
            return;
        }
        if (state.status === STATUS.SUCCESS && state.data) {
            renderSuccessState(state.data);
            return;
        }
        renderIdleState();
    }

    function getRequestPayload() {
        return {
            location: refs.locationInput.value.trim(),
            model: refs.modelSelect.value,
        };
    }

    return {
        refs,
        render,
        getRequestPayload,
    };
}
