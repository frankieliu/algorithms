// file:///C:/cygwin64/home/2025/algorithms/chrome/SurfingKeys/config.js
// settings.tabsThreshold = 0;

// Saving to a file
function saveTextToFile(text, filename) {
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href); // Clean up the object URL
}

const jsonData = {
    name: "John Doe",
    age: 30,
    city: "New York"
};

// Example usage:
// saveTextToFile('Hello, world!', 'my_document.txt');

function saveJsonToFile(data, filename) {
    const jsonString = JSON.stringify(data, null, 2); // Pretty-print JSON
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a); // Append to body (can be hidden)
    a.click();
    document.body.removeChild(a); // Clean up
    URL.revokeObjectURL(url);
}

// Call the function to save the JSON data
// saveJsonToFile(jsonData, 'my_data.json');

function timestampString() {
    const timeStamp = new Date().toLocaleString('en-CA', {
                    timeZone: 'PST',
                    hour12: false,
                    year: "numeric",
                    month: "2-digit",
                    day: "2-digit",
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                });
    const t2 = timeStamp.replace(/(\d+)[-.](\d+)[-.](\d+),\s(\d+):(\d+):(\d+)/, "$1$2$3_$4$5$6");
    return t2;
}

async function getTabs() {
    let otherTabs;
    let selfTabs;
    function a() {
        return new Promise((resolve, reject) => {
            api.RUNTIME('getTabs', { queryInfo: { currentWindow: false } }, response => {
                otherTabs = response;
                resolve("done a");
            });
        })
    };
    function b() {
        return new Promise((resolve, reject) => {
            api.RUNTIME('getTabs', { queryInfo: { currentWindow: true } }, response => {
                selfTabs = response;
                resolve("done b");
            });
        })
    };
    await Promise.all([a(), b()]);
    const allTabs = {
        "tabs": [...selfTabs.tabs, ...otherTabs.tabs]
    };
    return allTabs;
}

// an example to create a new mapping `ctrl-y`
api.mapkey('gw', 'Show me the money', async function () {
    const allTabs = await getTabs();
    console.log("Getting tabs ", allTabs);
    const formattedDate = timestampString();
    const prefix = "session";
    const filename = `${prefix}_${formattedDate}.json`;
    console.log("Saving to ", filename)
    saveJsonToFile(allTabs, filename);
});

// an example to replace `T` with `gt`, click `Default mappings` to see how `T` works.
api.map('gt', 'T');

api.map('g')

// an example to remove mapkey `Ctrl-i`
api.unmap('<ctrl-i>');

// set theme
settings.theme = `
.sk_theme {
    font-family: Input Sans Condensed, Charcoal, sans-serif;
    font-size: 10pt;
    background: #24272e;
    color: #abb2bf;
}
.sk_theme tbody {
    color: #fff;
}
.sk_theme input {
    color: #d0d0d0;
}
.sk_theme .url {
    color: #61afef;
}
.sk_theme .annotation {
    color: #56b6c2;
}
.sk_theme .omnibar_highlight {
    color: #528bff;
}
.sk_theme .omnibar_timestamp {
    color: #e5c07b;
}
.sk_theme .omnibar_visitcount {
    color: #98c379;
}
.sk_theme #sk_omnibarSearchResult ul li:nth-child(odd) {
    background: #303030;
}
.sk_theme #sk_omnibarSearchResult ul li.focused {
    background: #3e4452;
}
#sk_status, #sk_find {
    font-size: 20pt;
}`;
// click `Save` button to make above settings to take effect.</ctrl-i></ctrl-y>