document.addEventListener('DOMContentLoaded', initializeExplorer);

const FILE_STORAGE_KEY = 'virtualFileExplorerFiles';
let currentPath = '/';
let files = {}; // In-memory representation of our file system
let selectedItemId = null; // ID of the currently selected file/folder
let draggedItemId = null; // ID of the item being dragged

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

// Dialogs (unchanged, but still part of the HTML)
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
    currentPathSpan.textContent = getFullPath(currentPath); // Show full path now

    const children = getChildrenOfPath(currentPath);

    if (currentPath !== '/') {
        goBackBtn.style.display = 'inline-block';
    } else {
        goBackBtn.style.display = 'none';
    }

    if (children.length === 0) {
        fileListDiv.textContent = 'This folder is empty.';
        // Also make the fileListDiv a potential drop target if empty
        fileListDiv.addEventListener('dragover', handleDragOver);
        fileListDiv.addEventListener('dragleave', handleDragLeave);
        fileListDiv.addEventListener('drop', handleDropOnEmptyFolder);
    } else {
        // Remove listeners if it's not empty, they'll be on folder items
        fileListDiv.removeEventListener('dragover', handleDragOver);
        fileListDiv.removeEventListener('dragleave', handleDragLeave);
        fileListDiv.removeEventListener('drop', handleDropOnEmptyFolder);
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

        // Add drag-and-drop attributes
        itemDiv.setAttribute('draggable', true);
        itemDiv.addEventListener('dragstart', handleDragStart);
        itemDiv.addEventListener('dragend', handleDragEnd);

        if (item.type === 'folder') {
            itemDiv.addEventListener('dragover', handleDragOver);
            itemDiv.addEventListener('dragleave', handleDragLeave);
            itemDiv.addEventListener('drop', handleDrop);
        }


        itemDiv.addEventListener('click', (e) => {
            // Prevent selection when starting a drag
            if (!e.defaultPrevented) {
                selectItem(item.id);
            }
        });
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
    // For simplicity, this just deletes the direct children's entries in the 'files' object.
    // For a robust solution, you'd need a recursive function to delete all nested descendants.
    if (itemToDelete.type === 'folder' && itemToDelete.children) {
        itemToDelete.children.forEach(childId => {
            if (files[childId]) {
                // If it's a folder, ensure its children are also removed from 'files' recursively
                // (basic approach, consider a dedicated recursive delete function for deep deletion)
                if (files[childId].type === 'folder' && files[childId].children) {
                    files[childId].children.forEach(grandChildId => delete files[grandChildId]);
                }
                delete files[childId];
            }
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

    const itemToMove = files[selectedItemId];
    // Helper to check if a potential destination is a child of the item being moved
    const isDescendant = (potentialParentId, itemId) => {
        let current = files[potentialParentId];
        while (current) {
            if (current.id === itemId) return true;
            current = files[current.parent];
        }
        return false;
    };


    allFolders.forEach(folder => {
        // Prevent moving a folder into itself or its own subfolder
        if (itemToMove.type === 'folder' && isDescendant(folder.id, selectedItemId)) {
            return; // Skip if this folder is a descendant of the item being moved
        }
        // Also prevent moving to the current folder
        if (folder.id === itemToMove.parent) {
            return;
        }

        const option = document.createElement('option');
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
    if (itemId === '/') { // Special case for root
        segments.unshift('');
    } else if (segments.length === 0 && files[itemId]) { // If it's a top-level item that's not root, just its name
        segments.unshift(files[itemId].name);
    }
    return segments;
}

function getFullPath(itemId) {
    if (!files[itemId]) return '';
    const pathSegments = getPathSegments(itemId);
    // Join with '/' and handle leading slash for root properly
    return (itemId === '/' ? '/' : pathSegments.join('/'));
}


async function confirmMoveItem() {
    if (!selectedItemId) return;

    const destinationFolderId = moveFolderSelect.value;
    if (!destinationFolderId) {
        alert('Please select a destination folder.');
        return;
    }

    await performMove(selectedItemId, destinationFolderId);

    moveDialog.style.display = 'none';
    hideItemDetails(); // Hide details and re-render
    renderFileList();
}

/**
 * Performs the actual logic of moving an item from its current parent to a new parent.
 * @param {string} itemId The ID of the item to move.
 * @param {string} destinationFolderId The ID of the destination folder.
 */
async function performMove(itemId, destinationFolderId) {
    const itemToMove = files[itemId];
    const oldParent = files[itemToMove.parent];
    const newParent = files[destinationFolderId];

    if (!itemToMove || !oldParent || !newParent || newParent.type !== 'folder') {
        console.error('Move failed: Invalid item or folders.', {itemId, destinationFolderId, itemToMove, oldParent, newParent});
        return false;
    }

    // Check if the item is already in the destination folder
    if (itemToMove.parent === destinationFolderId) {
        console.log('Item is already in the target folder. No move needed.');
        return true; // Consider it a successful "no-op" move
    }

    // Prevent moving a parent folder into one of its children
    let tempParent = newParent;
    while(tempParent) {
        if (tempParent.id === itemId) {
            alert("Cannot move a folder into its own subfolder.");
            return false;
        }
        tempParent = files[tempParent.parent];
    }


    // Remove from old parent's children
    oldParent.children = oldParent.children.filter(id => id !== itemId);

    // Add to new parent's children
    newParent.children.push(itemId);
    itemToMove.parent = destinationFolderId;

    await saveFilesToStorage();
    return true;
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


// --- Drag and Drop Handlers ---

function handleDragStart(e) {
    draggedItemId = e.target.dataset.id;
    e.dataTransfer.setData('text/plain', draggedItemId); // Store the ID of the dragged item
    e.dataTransfer.effectAllowed = 'move';
    e.target.classList.add('dragging');
    console.log('Drag started for:', files[draggedItemId].name);
}

function handleDragOver(e) {
    e.preventDefault(); // Crucial: Allows a drop to happen
    const targetElement = e.target.closest('.folder-item, #file-list'); // Can drop on folder item or empty list
    if (targetElement) {
        const targetFolderId = targetElement.dataset.id || currentPath; // If #file-list, target is currentPath
        if (targetFolderId === draggedItemId) { // Cannot drop item onto itself
             e.dataTransfer.dropEffect = 'none'; // Indicate no drop allowed
             return;
        }

        // Prevent dropping a folder into its own subfolder
        if (files[draggedItemId] && files[draggedItemId].type === 'folder') {
            let potentialParent = files[targetFolderId];
            while (potentialParent) {
                if (potentialParent.id === draggedItemId) {
                    e.dataTransfer.dropEffect = 'none'; // Indicate no drop allowed
                    targetElement.classList.remove('drop-target'); // Remove highlight
                    return;
                }
                potentialParent = files[potentialParent.parent];
            }
        }


        e.dataTransfer.dropEffect = 'move'; // Visual feedback for 'move' operation
        if (targetFolderId !== draggedItemId) { // Don't highlight if dragging over itself
            targetElement.classList.add('drop-target');
        }
    }
}

function handleDragLeave(e) {
    const targetElement = e.target.closest('.folder-item, #file-list');
    if (targetElement) {
        targetElement.classList.remove('drop-target');
    }
}

async function handleDrop(e) {
    e.preventDefault(); // Prevent default browser behavior (like opening file)
    const targetElement = e.target.closest('.folder-item');
    const targetFolderId = targetElement ? targetElement.dataset.id : null;

    if (!draggedItemId || !targetFolderId) {
        console.warn('Drop failed: No item dragged or no valid target folder.');
        return;
    }

    if (draggedItemId === targetFolderId) {
        console.warn('Cannot drop item onto itself.');
        targetElement.classList.remove('drop-target');
        return;
    }

    // Also prevent dropping a folder into its own subfolder (redundant check but safe)
    if (files[draggedItemId] && files[draggedItemId].type === 'folder') {
        let potentialParent = files[targetFolderId];
        while (potentialParent) {
            if (potentialParent.id === draggedItemId) {
                console.warn('Cannot move a folder into its own subfolder.');
                targetElement.classList.remove('drop-target');
                return;
            }
            potentialParent = files[potentialParent.parent];
        }
    }


    console.log(`Attempting to move item ID ${draggedItemId} to folder ID ${targetFolderId}`);

    const success = await performMove(draggedItemId, targetFolderId);
    if (success) {
        hideItemDetails(); // Hide details as selection might change due to move
        renderFileList(); // Re-render the list to reflect changes
    }
    targetElement.classList.remove('drop-target'); // Clean up visual feedback
    draggedItemId = null; // Reset dragged item ID
}

async function handleDropOnEmptyFolder(e) {
    e.preventDefault(); // Prevent default browser behavior (like opening file)
    // This function is only attached to the fileListDiv when it's empty
    const targetFolderId = currentPath; // The current folder is the drop target

    if (!draggedItemId || !targetFolderId) {
        console.warn('Drop failed: No item dragged or no valid target folder (empty list drop).');
        return;
    }

    // Prevent dropping a folder into its own subfolder (redundant check but safe)
    if (files[draggedItemId] && files[draggedItemId].type === 'folder') {
        let potentialParent = files[targetFolderId];
        while (potentialParent) {
            if (potentialParent.id === draggedItemId) {
                console.warn('Cannot move a folder into its own subfolder.');
                fileListDiv.classList.remove('drop-target');
                return;
            }
            potentialParent = files[potentialParent.parent];
        }
    }

    console.log(`Attempting to move item ID ${draggedItemId} to current (empty) folder ID ${targetFolderId}`);

    const success = await performMove(draggedItemId, targetFolderId);
    if (success) {
        hideItemDetails();
        renderFileList();
    }
    fileListDiv.classList.remove('drop-target');
    draggedItemId = null;
}


function handleDragEnd(e) {
    // Clean up any lingering 'dragging' class
    const draggedElement = document.querySelector('.dragging');
    if (draggedElement) {
        draggedElement.classList.remove('dragging');
    }
    // Clean up any lingering 'drop-target' classes just in case
    document.querySelectorAll('.drop-target').forEach(el => el.classList.remove('drop-target'));
    draggedItemId = null; // Clear the dragged item ID
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