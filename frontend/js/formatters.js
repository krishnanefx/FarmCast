export function toCelsius(tempRaw) {
    return tempRaw > 100 ? tempRaw - 273.15 : tempRaw;
}

export function toHpa(pressureRaw) {
    return pressureRaw > 2000 ? pressureRaw / 100 : pressureRaw;
}

export function windMagnitude(uComp, vComp) {
    return Math.sqrt((uComp ** 2) + (vComp ** 2));
}

export function windDirectionDegrees(uComp, vComp) {
    return (Math.atan2(uComp, vComp) * (180 / Math.PI) + 180 + 360) % 360;
}

export function prettySource(source) {
    if (!source || !source.present) {
        return source?.warning || "No source metadata";
    }
    if (source.date || source.time) {
        return `${source.date || "unknown-date"} ${source.time || ""}`.trim();
    }
    return source.analysis_time || "unknown-analysis";
}

export function prettyAgeHours(source) {
    if (typeof source?.age_hours === "number") {
        return `${source.age_hours.toFixed(1)} hours`;
    }
    return "Unknown";
}

export function prettyUpdatedAt(source, fallbackForecastTime) {
    const raw = source?.fetched_at || source?.download_time || fallbackForecastTime;
    if (!raw) return "Unknown";
    return raw;
}
