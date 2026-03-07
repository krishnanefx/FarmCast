export const STATUS = Object.freeze({
    IDLE: "idle",
    LOADING: "loading",
    SUCCESS: "success",
    ERROR: "error",
});

export function createStateStore(initialState) {
    let state = { ...initialState };
    const listeners = new Set();

    function getState() {
        return state;
    }

    function setState(patch) {
        state = { ...state, ...patch };
        listeners.forEach((listener) => listener(state));
    }

    function subscribe(listener) {
        listeners.add(listener);
        return () => listeners.delete(listener);
    }

    return {
        getState,
        setState,
        subscribe,
    };
}
