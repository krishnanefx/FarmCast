import { fetchWeather } from "./api.js";
import { createRenderer } from "./render.js";
import { createStateStore, STATUS } from "./state.js";

document.addEventListener("DOMContentLoaded", () => {
    const store = createStateStore({
        status: STATUS.IDLE,
        data: null,
        error: null,
    });

    const renderer = createRenderer();
    store.subscribe(renderer.render);
    renderer.render(store.getState());

    renderer.refs.form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const payload = renderer.getRequestPayload();

        if (!payload.location) {
            store.setState({
                status: STATUS.ERROR,
                error: {
                    message: "Location is required.",
                    help: "Enter a city or region name before submitting.",
                },
            });
            return;
        }

        store.setState({
            status: STATUS.LOADING,
            data: null,
            error: null,
        });

        try {
            const response = await fetchWeather(payload);
            store.setState({
                status: STATUS.SUCCESS,
                data: response,
                error: null,
            });
        } catch (error) {
            store.setState({
                status: STATUS.ERROR,
                data: null,
                error: {
                    message: error?.message || "Unable to fetch weather prediction.",
                    help: error?.help,
                },
            });
        }
    });
});
