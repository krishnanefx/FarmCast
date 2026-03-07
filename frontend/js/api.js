const UK_REGIONS = [
    "East Midlands",
    "Yorkshire",
    "East Anglia",
    "Southwest England",
    "Wales",
    "Scotland",
    "Northern Ireland",
];

function toActionableHint(message = "") {
    const lower = message.toLowerCase();
    if (lower.includes("location not found")) {
        return "Try a more specific location format, such as 'Singapore, SG' or 'Nairobi, KE'.";
    }
    if (lower.includes("model") && lower.includes("not found")) {
        return "Model path is missing. Verify your pangu_weather_*.onnx files are present.";
    }
    if (lower.includes("input") && lower.includes("missing")) {
        return "ERA5 input files are missing. Refresh atmospheric inputs and retry.";
    }
    if (lower.includes("geocoder")) {
        return "Geocoding service failed. Retry in a moment or check network access.";
    }
    return "Check model files, source inputs, and API service health, then retry.";
}

async function requestWeather(payload) {
    const response = await fetch("/api/weather", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = body.detail || "Unable to fetch weather prediction.";
        const error = new Error(detail);
        error.help = toActionableHint(detail);
        error.status = response.status;
        throw error;
    }

    return body;
}

export async function fetchWeather({ location, model }) {
    return requestWeather({ location, model });
}

export async function fetchRegionalForecasts({ model }) {
    const calls = UK_REGIONS.map((region) =>
        requestWeather({ location: region, model })
    );
    const settled = await Promise.allSettled(calls);

    return settled.map((result, index) => {
        const region = UK_REGIONS[index];
        if (result.status === "fulfilled") {
            return { region, data: result.value, error: null };
        }
        return {
            region,
            data: null,
            error: result.reason?.message || "Regional forecast unavailable.",
        };
    });
}
