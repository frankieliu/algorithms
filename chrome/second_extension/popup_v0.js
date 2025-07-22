document.addEventListener('DOMContentLoaded', initializeExplorer);

const FILE_STORAGE_KEY = 'virtualFileExplorerFiles';
let currentPath = '/';
let files = {}; // In-memory representation of our file system
let selectedItemId = null; // ID of the currently selected file/folder

// --- UI Elements ---
const fileListDiv = document.getElementById('file-list');
const currentPathSpan = document.getElementById('current-path');
const createFolderBtn = document.getElementById('create-folder-btn');
const createFileBtn = document.getElementById('create-file-btn');
const importDownloadBtn = document.getElementById('import-download-btn');
const goBackBtn = document.getElementById('go-back-btn');

// File Details elements
const fileDetailsDiv = document.getElementById('file-details');
const detailNameSpan = document.getElementById('detail-name');
const detailTypeSpan = document.getElementById('detail-type');
const detailContentArea = document.getElementById('detail-content-area');
const detailContentInput = document.getElementById('detail-content');
const saveContentBtn = document.getElementById('save-content-btn');
const moveBtn = document.getElementById('move-btn');
const renameBtn = document.getElementById('rename-btn');
const deleteBtn = document.getElementById('delete-btn');
const closeDetailsBtn = document.getElementById('close-details-btn');

// Dialogs
const moveDialog = document.getElementById('move-dialog');
const moveItemNameSpan = document.getElementById('move-item-name');
const moveFolderSelect = document.getElementById('move-folder-select');
const confirmMoveBtn = document.getElementById('confirm-move-btn');
const cancelMoveBtn = document.getElementById('cancel-move-btn');

const renameDialog = document.getElementById('rename-dialog');
const renameOldNameSpan = document.getElementById('rename-old-name');
const renameNewNameInput = document.getElementById('rename-new-name');
const confirmRenameBtn = document.getElementById('confirm-rename-btn');
const cancelRenameBtn = document.getElementById('cancel-rename-btn');

const importDialog = document.getElementById('import-dialog');
const downloadSelect = document.getElementById('download-select');
const confirmImportBtn = document.getElementById('confirm-import-btn');
const cancelImportBtn = document.getElementById('cancel-import-btn');

// --- Helper Functions ---

function generateUniqueId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 7);
}

async function saveFilesToStorage() {
    await chrome.storage.local.set({ [FILE_STORAGE_KEY]: files });
}

async function loadFilesFromStorage() {
    const result = await chrome.storage.local.get(FILE_STORAGE_KEY);
    files = result[FILE_STORAGE_KEY] || {
        '/': { id: '/', name: '/', type: 'folder', parent: null, children: [] }
    };
    // Ensure root exists if it was somehow deleted
    if (!files['/']) {
        files['/'] = { id: '/', name: '/', type: 'folder', parent: null, children: [] };
    }
}

function getChildrenOfPath(path) {
    const parentFolder = files[path];
    if (!parentFolder || parentFolder.type !== 'folder') {
        return [];
    }
    return parentFolder.children.map(id => files[id]).filter(Boolean).sort((a, b) => {
        if (a.type === 'folder' && b.type !== 'folder') return -1;
        if (a.type !== 'folder' && b.type === 'folder') return 1;
        return a.name.localeCompare(b.name);
    });
}

function renderFileList() {
    fileListDiv.innerHTML = '';
    currentPathSpan.textContent = currentPath;

    const children = getChildrenOfPath(currentPath);

    if (currentPath !== '/') {
        goBackBtn.style.display = 'inline-block';
    } else {
        goBackBtn.style.display = 'none';
    }

    if (children.length === 0) {
        fileListDiv.textContent = 'This folder is empty.';
        return;
    }

    children.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.classList.add('file-item');
        if (item.type === 'folder') {
            itemDiv.classList.replace('file-item', 'folder-item');
        }
        if (selectedItemId === item.id) {
            itemDiv.classList.add('selected');
        }

        const iconSpan = document.createElement('span');
        iconSpan.classList.add('item-icon');
        iconSpan.textContent = item.type === 'folder' ? '📁' : (item.type === 'note' ? '📄' : '📎'); // Simple icons

        const nameSpan = document.createElement('span');
        nameSpan.textContent = item.name;

        itemDiv.appendChild(iconSpan);
        itemDiv.appendChild(nameSpan);
        itemDiv.dataset.id = item.id;
        itemDiv.dataset.type = item.type;

        itemDiv.addEventListener('click', () => selectItem(item.id));
        if (item.type === 'folder') {
            itemDiv.addEventListener('dblclick', () => navigateIntoFolder(item.id));
        }

        fileListDiv.appendChild(itemDiv);
    });
}

function selectItem(id) {
    if (selectedItemId) {
        const prevSelected = fileListDiv.querySelector(`[data-id="${selectedItemId}"]`);
        if (prevSelected) prevSelected.classList.remove('selected');
    }
    selectedItemId = id;
    const currentSelected = fileListDiv.querySelector(`[data-id="${selectedItemId}"]`);
    if (currentSelected) currentSelected.classList.add('selected');

    displayItemDetails(id);
}

function displayItemDetails(id) {
    const item = files[id];
    if (!item) {
        fileDetailsDiv.style.display = 'none';
        return;
    }

    fileDetailsDiv.style.display = 'block';
    detailNameSpan.textContent = item.name;
    detailTypeSpan.textContent = item.type;

    if (item.type === 'note') {
        detailContentArea.style.display = 'block';
        detailContentInput.value = item.content || '';
    } else {
        detailContentArea.style.display = 'none';
        detailContentInput.value = '';
    }
}

function hideItemDetails() {
    fileDetailsDiv.style.display = 'none';
    selectedItemId = null;
    renderFileList(); // Unselect any item in the list
}

// --- File System Operations ---

async function createFolder() {
    const folderName = prompt('Enter folder name:');
    if (folderName) {
        const newFolderId = generateUniqueId();
        files[newFolderId] = {
            id: newFolderId,
            name: folderName,
            type: 'folder',
            parent: currentPath,
            children: []
        };
        files[currentPath].children.push(newFolderId);
        await saveFilesToStorage();
        renderFileList();
    }
}

async function createFile(type = 'note', content = '') {
    const fileName = prompt('Enter file name:');
    if (fileName) {
        const newFileId = generateUniqueId();
        files[newFileId] = {
            id: newFileId,
            name: fileName,
            type: type,
            parent: currentPath,
            content: content // Only for 'note' type
        };
        files[currentPath].children.push(newFileId);
        await saveFilesToStorage();
        renderFileList();
    }
}

async function saveFileContent() {
    if (selectedItemId && files[selectedItemId] && files[selectedItemId].type === 'note') {
        files[selectedItemId].content = detailContentInput.value;
        await saveFilesToStorage();
        alert('Content saved!');
    }
}

async function deleteItem() {
    if (!selectedItemId) return;

    const itemToDelete = files[selectedItemId];
    if (!itemToDelete) return;

    if (!confirm(`Are you sure you want to delete "${itemToDelete.name}"?`)) {
        return;
    }

    // Remove from parent's children array
    const parent = files[itemToDelete.parent];
    if (parent && parent.children) {
        parent.children = parent.children.filter(id => id !== itemToDelete.id);
    }

    // Recursively delete children if it's a folder
    if (itemToDelete.type === 'folder' && itemToDelete.children) {
        itemToDelete.children.forEach(childId => {
            // This is a simple deletion, for a real app, you'd want a more robust recursive delete function
            delete files[childId];
        });
    }

    delete files[selectedItemId];
    selectedItemId = null; // Deselect
    await saveFilesToStorage();
    renderFileList();
    hideItemDetails();
}

async function renameItem() {
    if (!selectedItemId) return;
    const item = files[selectedItemId];
    if (!item) return;

    renameOldNameSpan.textContent = item.name;
    renameNewNameInput.value = item.name;
    renameDialog.style.display = 'block';

    confirmRenameBtn.onclick = async () => {
        const newName = renameNewNameInput.value.trim();
        if (newName && newName !== item.name) {
            item.name = newName;
            await saveFilesToStorage();
            renderFileList();
            displayItemDetails(selectedItemId); // Update details with new name
        }
        renameDialog.style.display = 'none';
    };

    cancelRenameBtn.onclick = () => {
        renameDialog.style.display = 'none';
    };
}

async function openMoveDialog() {
    if (!selectedItemId) return;

    moveItemNameSpan.textContent = files[selectedItemId].name;
    moveFolderSelect.innerHTML = ''; // Clear previous options

    // Populate dropdown with folders (excluding the item itself and its descendants if it's a folder)
    const allFolders = Object.values(files).filter(item => item.type === 'folder' && item.id !== selectedItemId);

    // Prevent moving a folder into its own subfolder
    const itemToMove = files[selectedItemId];
    const itemPathSegments = getPathSegments(selectedItemId); // Get the full path segments of the item
    const currentPathSegments = getPathSegments(currentPath);

    allFolders.forEach(folder => {
        // Skip current path and child paths if moving a folder
        const folderPathSegments = getPathSegments(folder.id);
        if (itemToMove.type === 'folder' && folderPathSegments.length > itemPathSegments.length &&
            folderPathSegments.slice(0, itemPathSegments.length).every((seg, i) => seg === itemPathSegments[i])) {
            return; // This folder is a descendant of the item being moved
        }

        const option = document.createElement('option');
        // Display full path for better clarity in move dialog
        option.value = folder.id;
        option.textContent = getFullPath(folder.id);
        moveFolderSelect.appendChild(option);
    });

    moveDialog.style.display = 'block';
}

function getPathSegments(itemId) {
    const segments = [];
    let current = files[itemId];
    while (current && current.id !== '/') {
        segments.unshift(current.name);
        current = files[current.parent];
    }
    segments.unshift(''); // For the leading '/'
    return segments;
}

function getFullPath(itemId) {
    return getPathSegments(itemId).join('/');
}


async function confirmMoveItem() {
    if (!selectedItemId) return;

    const destinationFolderId = moveFolderSelect.value;
    if (!destinationFolderId) {
        alert('Please select a destination folder.');
        return;
    }

    const itemToMove = files[selectedItemId];
    const oldParent = files[itemToMove.parent];
    const newParent = files[destinationFolderId];

    if (!oldParent || !newParent) {
        alert('Error: Parent folders not found.');
        return;
    }

    // Remove from old parent's children
    oldParent.children = oldParent.children.filter(id => id !== selectedItemId);

    // Add to new parent's children
    newParent.children.push(selectedItemId);
    itemToMove.parent = destinationFolderId;

    await saveFilesToStorage();
    moveDialog.style.display = 'none';
    hideItemDetails(); // Hide details and re-render
    renderFileList();
}

// --- Navigation ---

function navigateIntoFolder(folderId) {
    if (files[folderId] && files[folderId].type === 'folder') {
        currentPath = folderId;
        hideItemDetails(); // Hide details when navigating
        renderFileList();
    }
}

function goBack() {
    if (currentPath !== '/') {
        const currentFolder = files[currentPath];
        if (currentFolder && currentFolder.parent) {
            currentPath = currentFolder.parent;
            hideItemDetails(); // Hide details when navigating
            renderFileList();
        }
    }
}

// --- Downloads Integration ---

async function openImportDialog() {
    downloadSelect.innerHTML = '';
    const downloads = await chrome.downloads.search({}); // Get all downloads

    if (downloads.length === 0) {
        const option = document.createElement('option');
        option.textContent = 'No downloads found.';
        option.disabled = true;
        downloadSelect.appendChild(option);
    } else {
        downloads.forEach(dl => {
            if (dl.filename) { // Ensure filename exists
                const option = document.createElement('option');
                option.value = dl.id;
                option.textContent = dl.filename.split(/[\\/]/).pop(); // Get just the file name
                downloadSelect.appendChild(option);
            }
        });
    }
    importDialog.style.display = 'block';
}

async function importDownload() {
    const selectedDownloadId = downloadSelect.value;
    if (!selectedDownloadId) {
        alert('Please select a download to import.');
        return;
    }

    const downloads = await chrome.downloads.search({ id: parseInt(selectedDownloadId) });
    const download = downloads[0];

    if (download) {
        // Create a virtual file entry for the downloaded file
        const newFileId = generateUniqueId();
        files[newFileId] = {
            id: newFileId,
            name: download.filename.split(/[\\/]/).pop(), // Use actual file name
            type: 'download',
            parent: currentPath,
            downloadId: download.id, // Store the actual Chrome download ID
            downloadPath: download.filename // Store the actual system path (for display)
        };
        files[currentPath].children.push(newFileId);
        await saveFilesToStorage();
        renderFileList();
        importDialog.style.display = 'none';
        alert(`'${download.filename.split(/[\\/]/).pop()}' imported virtually!`);
    } else {
        alert('Selected download not found.');
    }
}

// --- Event Listeners ---
createFolderBtn.addEventListener('click', createFolder);
createFileBtn.addEventListener('click', () => createFile('note'));
importDownloadBtn.addEventListener('click', openImportDialog);
goBackBtn.addEventListener('click', goBack);

saveContentBtn.addEventListener('click', saveFileContent);
moveBtn.addEventListener('click', openMoveDialog);
renameBtn.addEventListener('click', renameItem);
deleteBtn.addEventListener('click', deleteItem);
closeDetailsBtn.addEventListener('click', hideItemDetails);

confirmMoveBtn.addEventListener('click', confirmMoveItem);
cancelMoveBtn.addEventListener('click', () => moveDialog.style.display = 'none');

confirmImportBtn.addEventListener('click', importDownload);
cancelImportBtn.addEventListener('click', () => importDialog.style.display = 'none');

// --- Initialization ---
async function initializeExplorer() {
    await loadFilesFromStorage();
    renderFileList();
}