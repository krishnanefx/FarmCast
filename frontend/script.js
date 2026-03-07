document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('weather-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');

    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    // Result elements
    const resLocation = document.getElementById('result-location');
    const resCoords = document.getElementById('result-coords');
    const resSource = document.getElementById('result-source');
    const metTemp = document.getElementById('metric-temp');
    const metWind = document.getElementById('metric-wind');
    const metPressure = document.getElementById('metric-pressure');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const location = document.getElementById('location').value;
        const model = document.getElementById('model-select').value;

        // Reset UI
        resultsContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');

        // Loading state
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/weather', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ location, model })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An error occurred while fetching the weather prediction.');
            }

            let temp = data.metrics.temperature_2m;
            if (temp > 100) {
                temp = temp - 273.15; // Convert Kelvin down to Celsius
            }

            let pressure = data.metrics.mean_sea_level_pressure;
            if (pressure > 2000) {
                pressure = pressure / 100; // Convert Pa down to hPa
            }

            const windSpeed = Math.sqrt(
                Math.pow(data.metrics.u_wind_10m, 2) + Math.pow(data.metrics.v_wind_10m, 2)
            );

            // Update DOM
            resLocation.textContent = data.location;
            resCoords.textContent = `Lat: ${data.coordinates.lat.toFixed(4)}, Lon: ${data.coordinates.lon.toFixed(4)} | Grid: [${data.grid_idx.lat_idx}, ${data.grid_idx.lon_idx}]`;
            const src = (() => {
                if (!data.source || !data.source.present) {
                    return `Source: ${data.source && data.source.warning ? data.source.warning : 'No source metadata'}`;
                }
                if (data.source.date || data.source.time) {
                    const age = typeof data.source.age_hours === 'number' ? `, age ${data.source.age_hours}h` : '';
                    return `Source: ${data.source.date || 'unknown-date'} ${data.source.time || ''} (${data.source.fetched_at || 'no fetch timestamp'}${age})`;
                }
                const age = typeof data.source.age_hours === 'number' ? `, age ${data.source.age_hours}h` : '';
                return `Source: ${data.source.analysis_time || 'unknown-analysis'} (${data.source.download_time || 'no download timestamp'}${age})`;
            })();
            resSource.textContent = src;

            metTemp.textContent = `${temp.toFixed(1)} °C`;
            metWind.textContent = `${windSpeed.toFixed(1)} m/s`;
            metPressure.textContent = `${Math.round(pressure)} hPa`;

            // Show results
            resultsContainer.classList.remove('hidden');

            // Small transition effect
            resultsContainer.style.opacity = 0;
            setTimeout(() => {
                resultsContainer.style.transition = 'opacity 0.5s ease-in';
                resultsContainer.style.opacity = 1;
            }, 50);

        } catch (error) {
            errorMessage.textContent = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            // Restore button
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
});
