import { fetchRegionalForecasts, fetchWeather } from "./api.js";
import { createRenderer } from "./render.js";
import { createStateStore, STATUS } from "./state.js";

document.addEventListener("DOMContentLoaded", () => {
    const store = createStateStore({
        status: STATUS.IDLE,
        data: null,
        error: null,
        lastAction: "predict",
        lastPayload: null,
    });

    const renderer = createRenderer();
    store.subscribe(renderer.render);
    renderer.render(store.getState());

    async function runForecast(payload, action = "predict") {
        if (!payload.location) {
            store.setState({
                status: STATUS.ERROR,
                data: null,
                error: {
                    message: "Location is required.",
                    help: "Enter a city or region name before submitting.",
                },
                lastAction: action,
            });
            return;
        }

        store.setState({
            status: STATUS.LOADING,
            data: null,
            error: null,
            lastAction: action,
        });

        try {
            const [response, regional] = await Promise.all([
                fetchWeather(payload),
                fetchRegionalForecasts({ model: payload.model }),
            ]);

            store.setState({
                status: STATUS.SUCCESS,
                data: {
                    ...response,
                    regional,
                },
                error: null,
                lastAction: action,
                lastPayload: payload,
            });
        } catch (error) {
            store.setState({
                status: STATUS.ERROR,
                data: null,
                error: {
                    message: error?.message || "Unable to fetch weather prediction.",
                    help: error?.help,
                },
                lastAction: action,
            });
        }
    }

    renderer.refs.form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const payload = renderer.getRequestPayload();
        await runForecast(payload, "predict");
    });

    renderer.refs.refreshBtn.addEventListener("click", async () => {
        const payload = renderer.getRefreshPayload(store.getState().lastPayload);
        await runForecast(payload, "refresh");
    });
});
