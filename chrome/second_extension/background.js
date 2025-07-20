// background.js (Service Worker)

let explorerTabId = null; // Variable to store the ID of our explorer tab

// Listen for when the extension's action (toolbar icon) is clicked
chrome.action.onClicked.addListener(async (tab) => {
    const explorerUrl = chrome.runtime.getURL("popup.html"); // Get the full URL to our HTML page

    // Check if our explorer tab is already open
    if (explorerTabId !== null) {
        try {
            const existingTab = await chrome.tabs.get(explorerTabId);
            if (existingTab) {
                // Tab exists, update it to focus and navigate if needed
                await chrome.tabs.update(explorerTabId, { active: true, url: explorerUrl });
                return; // Stop here, no need to create a new tab
            }
        } catch (error) {
            // Tab ID might be invalid (e.g., tab was closed), so reset it
            console.warn(`Explorer tab with ID ${explorerTabId} not found or error:`, error);
            explorerTabId = null;
        }
    }

    // If the tab is not open or its ID was invalid, create a new one
    try {
        const newTab = await chrome.tabs.create({ url: explorerUrl });
        explorerTabId = newTab.id; // Store the ID of the newly created tab
        console.log(`Virtual File Explorer opened in new tab with ID: ${explorerTabId}`);
    } catch (error) {
        console.error("Failed to open Virtual File Explorer tab:", error);
    }
});

// Optional: Listen for when tabs are closed to reset explorerTabId
chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
    if (tabId === explorerTabId) {
        explorerTabId = null; // Reset the ID if our explorer tab is closed
        console.log("Virtual File Explorer tab closed. Resetting tab ID.");
    }
});