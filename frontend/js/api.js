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

export async function fetchWeather({ location, model }) {
    const response = await fetch("/api/weather", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ location, model }),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = payload.detail || "Unable to fetch weather prediction.";
        const error = new Error(detail);
        error.help = toActionableHint(detail);
        error.status = response.status;
        throw error;
    }

    return payload;
}
