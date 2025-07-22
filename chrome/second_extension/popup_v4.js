document.addEventListener('DOMContentLoaded', initializeExplorer);

const FILE_STORAGE_KEY = 'virtualFileExplorerFiles';
const EXPANDED_FOLDERS_KEY = 'virtualFileExplorerExpandedFolders';

let currentPath = '/'; // Still tracking the 'current' folder for new creations
let files = {}; // In-memory representation of our file system
let expandedFolders = {}; // Store which folders are expanded (id: true/false)
let selectedItemId = null; // ID of the currently selected file/folder
let draggedItemId = null; // ID of the item being dragged

// --- UI Elements ---
const fileListDiv = document.getElementById('file-list');
const currentPathSpan = document.getElementById('current-path');
const createFolderBtn = document.getElementById('create-folder-btn');
const createFileBtn = document.getElementById('create-file-btn');
const importDownloadBtn = document.getElementById('import-download-btn');
const goBackBtn = document.getElementById('go-back-btn'); // Will be less critical with tree view but still useful

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
    const result = await chrome.storage.local.get([FILE_STORAGE_KEY, EXPANDED_FOLDERS_KEY]);
    files = result[FILE_STORAGE_KEY] || {
        '/': { id: '/', name: 'Root', type: 'folder', parent: null, children: [] }
    };
    expandedFolders = result[EXPANDED_FOLDERS_KEY] || { '/': true }; // Root is expanded by default

    // Ensure root exists if it was somehow deleted
    if (!files['/']) {
        files['/'] = { id: '/', name: 'Root', type: 'folder', parent: null, children: [] };
    }
}

async function saveExpandedFolders() {
    await chrome.storage.local.set({ [EXPANDED_FOLDERS_KEY]: expandedFolders });
}

/**
 * Recursively renders the file and folder hierarchy.
 * @param {string} parentId The ID of the current parent folder to render children from.
 * @param {number} level The current indentation level.
 * @param {HTMLElement} targetElement The HTML element to append children to.
 */
function renderHierarchy(parentId, level, targetElement) {
    const children = files[parentId].children
        .map(id => files[id])
        .filter(Boolean)
        .sort((a, b) => {
            if (a.type === 'folder' && b.type !== 'folder') return -1;
            if (a.type !== 'folder' && b.type === 'folder') return 1;
            return a.name.localeCompare(b.name);
        });

    const indentationPx = level * 15; // 15px per level

    children.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.classList.add(item.type === 'folder' ? 'folder-item' : 'file-item');
        itemDiv.style.setProperty('--indentation-level', `${indentationPx}px`); // Apply indentation
        if (selectedItemId === item.id) {
            itemDiv.classList.add('selected');
        }

        // Draggable attributes for ALL items
        itemDiv.setAttribute('draggable', true);
        itemDiv.addEventListener('dragstart', handleDragStart);
        itemDiv.addEventListener('dragend', handleDragEnd);

        // Drop target for folders only
        if (item.type === 'folder') {
            itemDiv.addEventListener('dragover', handleDragOverItem); // Specific item dragover
            itemDiv.addEventListener('dragleave', handleDragLeaveItem); // Specific item dragleave
            itemDiv.addEventListener('drop', handleDropItem); // Specific item drop
        }

        // Toggle icon for folders
        if (item.type === 'folder') {
            const toggleSpan = document.createElement('span');
            toggleSpan.classList.add('folder-toggle');
            toggleSpan.textContent = expandedFolders[item.id] ? '▼' : '▶'; // Down or Right arrow
            toggleSpan.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent folder from being selected/opened
                toggleFolder(item.id);
            });
            itemDiv.appendChild(toggleSpan);
        } else {
            // For files, add an empty space to align with folder content
            const emptyToggleSpace = document.createElement('span');
            emptyToggleSpace.classList.add('folder-toggle'); // Use the same class for spacing
            emptyToggleSpace.textContent = '';
            itemDiv.appendChild(emptyToggleSpace);
        }

        const iconSpan = document.createElement('span');
        iconSpan.classList.add('item-icon');
        iconSpan.textContent = item.type === 'folder' ? '📁' : (item.type === 'note' ? '📄' : '📎');

        const nameSpan = document.createElement('span');
        nameSpan.textContent = item.name;

        itemDiv.appendChild(iconSpan);
        itemDiv.appendChild(nameSpan);
        itemDiv.dataset.id = item.id;
        itemDiv.dataset.type = item.type;


        itemDiv.addEventListener('click', (e) => {
            if (!e.defaultPrevented) { // Check if default action (like drag) was prevented
                selectItem(item.id);
            }
        });

        targetElement.appendChild(itemDiv);

        // Recursively render children if the folder is expanded
        if (item.type === 'folder' && expandedFolders[item.id]) {
            renderHierarchy(item.id, level + 1, targetElement);
        }
    });
}

async function renderFileList() {
    fileListDiv.innerHTML = ''; // Clear existing list
    currentPathSpan.textContent = getFullPath(currentPath); // Path bar still useful

    // Always render from root for the tree view
    if (files['/']) {
        renderHierarchy('/', 0, fileListDiv);
    } else {
        fileListDiv.textContent = 'No root folder found! This is an error.';
    }

    // Always ensure the main file list area can accept drops if it represents the current (root) directory
    // or when nothing specific is highlighted.
    // Attach these to the fileListDiv (the main container) for dropping into the current folder
    fileListDiv.removeEventListener('dragover', handleDragOverGeneral);
    fileListDiv.removeEventListener('dragleave', handleDragLeaveGeneral);
    fileListDiv.removeEventListener('drop', handleDropGeneral);

    fileListDiv.addEventListener('dragover', handleDragOverGeneral);
    fileListDiv.addEventListener('dragleave', handleDragLeaveGeneral);
    fileListDiv.addEventListener('drop', handleDropGeneral);

    // Hide details if the selected item no longer exists (e.g., was deleted or moved)
    if (selectedItemId && !files[selectedItemId]) {
        hideItemDetails();
    }
}

async function toggleFolder(folderId) {
    expandedFolders[folderId] = !expandedFolders[folderId];
    await saveExpandedFolders();
    renderFileList(); // Re-render the entire list to reflect the change
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
    // Don't call renderFileList here, as it's often called after a file operation anyway.
    // If needed specifically to deselect, ensure no re-render loop.
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
        // Expand the parent folder if not already, so the new folder is visible
        if (!expandedFolders[currentPath]) {
            expandedFolders[currentPath] = true;
            await saveExpandedFolders();
        }
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
        // Expand the parent folder if not already, so the new file is visible
        if (!expandedFolders[currentPath]) {
            expandedFolders[currentPath] = true;
            await saveExpandedFolders();
        }
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

    if (!confirm(`Are you sure you want to delete "${itemToDelete.name}"? This will delete all its contents.`)) {
        return;
    }

    // Recursive deletion helper
    function deleteRecursive(itemIdToDelete) {
        const item = files[itemIdToDelete];
        if (!item) return;

        // If it's a folder, recursively delete its children
        if (item.type === 'folder' && item.children) {
            // Make a copy of children array to avoid modifying while iterating
            [...item.children].forEach(childId => deleteRecursive(childId));
        }
        // Delete the item itself
        delete files[itemIdToDelete];
        delete expandedFolders[itemIdToDelete]; // Also remove from expanded state
    }

    // Remove from parent's children array first
    const parent = files[itemToDelete.parent];
    if (parent && parent.children) {
        parent.children = parent.children.filter(id => id !== itemToDelete.id);
    }

    // Now perform the recursive deletion
    deleteRecursive(selectedItemId);

    selectedItemId = null; // Deselect
    await saveFilesToStorage();
    await saveExpandedFolders();
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

    const itemToMove = files[selectedItemId];

    // Get all folders, excluding the item itself and its descendants if it's a folder
    const allFolders = Object.values(files).filter(item => item.type === 'folder');

    allFolders.forEach(folder => {
        // Prevent moving a folder into itself or its own subfolder
        if (folder.id === selectedItemId || isDescendant(folder.id, selectedItemId)) {
            return;
        }
        // Prevent moving to the current parent folder
        if (folder.id === itemToMove.parent) {
             return;
        }

        const option = document.createElement('option');
        option.value = folder.id;
        option.textContent = getFullPath(folder.id);
        moveFolderSelect.appendChild(option);
    });

    if (moveFolderSelect.options.length === 0) {
        const option = document.createElement('option');
        option.textContent = 'No other folders available to move to.';
        option.disabled = true;
        moveFolderSelect.appendChild(option);
        confirmMoveBtn.disabled = true;
    } else {
        confirmMoveBtn.disabled = false;
    }

    moveDialog.style.display = 'block';
}

function isDescendant(potentialParentId, itemId) {
    let current = files[potentialParentId];
    while (current) {
        if (current.id === itemId) return true;
        current = files[current.parent];
    }
    return false;
}

function getPathSegments(itemId) {
    const segments = [];
    let current = files[itemId];
    // Traverse up to the root, collecting names
    while (current && current.parent !== null) { // Stop at root's parent (null)
        segments.unshift(current.name);
        current = files[current.parent];
    }
    // If current is the root itself, add its name
    if (current && current.id === '/') {
        segments.unshift(current.name);
    }
    return segments;
}

function getFullPath(itemId) {
    if (!files[itemId]) return '';
    const pathSegments = getPathSegments(itemId);
    // Handle root path separately to display as '/'
    if (itemId === '/') {
        return '/';
    }
    // For other items, join segments with '/' and ensure leading slash
    return '/' + pathSegments.join('/');
}

async function confirmMoveItem() {
    if (!selectedItemId) return;

    const destinationFolderId = moveFolderSelect.value;
    if (!destinationFolderId || confirmMoveBtn.disabled) { // Check if dropdown is empty or button disabled
        alert('Please select a valid destination folder.');
        return;
    }

    const success = await performMove(selectedItemId, destinationFolderId);
    if (success) {
        moveDialog.style.display = 'none';
        hideItemDetails();
        renderFileList();
    }
}

/**
 * Performs the actual logic of moving an item from its current parent to a new parent.
 * @param {string} itemId The ID of the item to move.
 * @param {string} destinationFolderId The ID of the destination folder.
 * @returns {boolean} True if move was successful, false otherwise.
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
    if (itemToMove.type === 'folder' && isDescendant(destinationFolderId, itemId)) {
        alert("Cannot move a folder into its own subfolder.");
        return false;
    }

    // Crucial: Remove from old parent's children array
    oldParent.children = oldParent.children.filter(id => id !== itemId);
    console.log(`Removed ${itemToMove.name} from old parent ${oldParent.name}. New children:`, oldParent.children);

    // Crucial: Add to new parent's children array
    newParent.children.push(itemId);
    itemToMove.parent = destinationFolderId;
    console.log(`Added ${itemToMove.name} to new parent ${newParent.name}. New children:`, newParent.children);

    await saveFilesToStorage();
    // Expand the new parent folder if it's not already, so the moved item is visible
    if (!expandedFolders[destinationFolderId]) {
        expandedFolders[destinationFolderId] = true;
        await saveExpandedFolders();
    }

    return true;
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
        confirmImportBtn.disabled = true;
    } else {
        downloads.forEach(dl => {
            if (dl.filename) { // Ensure filename exists
                const option = document.createElement('option');
                option.value = dl.id;
                option.textContent = dl.filename.split(/[\\/]/).pop(); // Get just the file name
                downloadSelect.appendChild(option);
            }
        });
        confirmImportBtn.disabled = false;
    }
    importDialog.style.display = 'block';
}

async function importDownload() {
    const selectedDownloadId = downloadSelect.value;
    if (!selectedDownloadId || confirmImportBtn.disabled) {
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
        // Expand the parent folder if not already, so the new file is visible
        if (!expandedFolders[currentPath]) {
            expandedFolders[currentPath] = true;
            await saveExpandedFolders();
        }
        alert(`'${download.filename.split(/[\\/]/).pop()}' imported virtually!`);
    } else {
        alert('Selected download not found.');
    }
}


// --- Drag and Drop Handlers ---
// Note: We use specific handlers for drag-over/leave/drop on individual items vs. the general list area

function handleDragStart(e) {
    draggedItemId = e.target.dataset.id;
    e.dataTransfer.setData('text/plain', draggedItemId); // Store the ID of the dragged item
    e.dataTransfer.effectAllowed = 'move';
    e.target.classList.add('dragging');
    console.log('Drag started for:', files[draggedItemId].name);
}

// Handler for drag-over specifically on a folder item (potential drop target)
function handleDragOverItem(e) {
    e.preventDefault(); // Crucial: Allows a drop to happen

    // *** CRITICAL FIX HERE ***
    // Ensure we are getting the ID from the actual folder element, not its children
    const targetElement = e.target.closest('.folder-item');
    if (!targetElement) { // Not over a folder item
        e.dataTransfer.dropEffect = 'none';
        return;
    }
    const targetFolderId = targetElement.dataset.id;
    // *** END CRITICAL FIX ***

    if (!draggedItemId) { // No item is currently being dragged
        e.dataTransfer.dropEffect = 'none';
        targetElement.classList.remove('drop-target');
        return;
    }

    const itemToMove = files[draggedItemId];

    // Do not allow dropping onto self
    if (targetFolderId === draggedItemId) {
        e.dataTransfer.dropEffect = 'none';
        targetElement.classList.remove('drop-target');
        return;
    }

    // Do not allow dropping if the target is the item's current parent
    if (itemToMove && itemToMove.parent === targetFolderId) {
         e.dataTransfer.dropEffect = 'none';
         targetElement.classList.remove('drop-target');
         return;
    }

    // Prevent dropping a folder into its own subfolder
    if (itemToMove.type === 'folder' && isDescendant(targetFolderId, draggedItemId)) {
        e.dataTransfer.dropEffect = 'none'; // Indicate no drop allowed
        targetElement.classList.remove('drop-target'); // Remove highlight
        return;
    }

    e.dataTransfer.dropEffect = 'move'; // Visual feedback for 'move' operation
    targetElement.classList.add('drop-target');
}

// Handler for drag-leave specifically from a folder item
function handleDragLeaveItem(e) {
    const targetElement = e.target.closest('.folder-item');
    if (targetElement) {
        targetElement.classList.remove('drop-target');
    }
}

// Handler for drop specifically on a folder item
async function handleDropItem(e) {
    e.preventDefault(); // Prevent default browser behavior (like opening file)

    // *** CRITICAL FIX HERE ***
    // Ensure we are getting the ID from the actual folder element, not its children
    const targetElement = e.target.closest('.folder-item');
    const targetFolderId = targetElement ? targetElement.dataset.id : null;
    // *** END CRITICAL FIX ***

    if (!draggedItemId || !targetFolderId) {
        console.warn('Drop failed: No item dragged or no valid target folder.');
        if (targetElement) targetElement.classList.remove('drop-target'); // Clean up if it was highlighted
        return;
    }

    const itemToMove = files[draggedItemId];

    // Re-check conditions at drop for safety, though dragOver should handle most
    if (targetFolderId === draggedItemId ||
        itemToMove.parent === targetFolderId ||
        (itemToMove.type === 'folder' && isDescendant(targetFolderId, draggedItemId))) {
        console.warn('Invalid drop target detected during drop event.');
        if (targetElement) targetElement.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    console.log(`Attempting to move item ID ${draggedItemId} to folder ID ${targetFolderId}`);

    const success = await performMove(draggedItemId, targetFolderId);
    if (success) {
        hideItemDetails(); // Hide details as selection might change due to move
        renderFileList(); // Re-render the list to reflect changes
    }
    if (targetElement) targetElement.classList.remove('drop-target'); // Clean up visual feedback
    draggedItemId = null; // Reset dragged item ID
}


// Handler for dropping an item onto the general fileListDiv area (implies dropping into the *root* folder, or currentPath if we were tracking it differently)
function handleDragOverGeneral(e) {
    e.preventDefault();

    // *** CRITICAL FIX HERE ***
    // Ensure draggedItemId is available and the item exists
    if (!draggedItemId || !files[draggedItemId]) {
        e.dataTransfer.dropEffect = 'none';
        fileListDiv.classList.remove('drop-target'); // Ensure no lingering highlight
        return;
    }
    // *** END CRITICAL FIX ***

    const itemToMove = files[draggedItemId];
    // This `handleDropGeneral` is specifically for dropping into the *root* folder.
    // If you want to drop into the `currentPath` *always* when dragging onto blank space,
    // you would use `currentPath` here instead of `'/'`.
    const targetRootId = '/';

    // Allow drop on general list area if dropping into the root folder (or currentPath)
    // AND it's not the item's current parent AND not trying to move a folder into its descendant
    if (itemToMove.parent !== targetRootId && !(itemToMove.type === 'folder' && isDescendant(targetRootId, draggedItemId))) {
        e.dataTransfer.dropEffect = 'move';
        fileListDiv.classList.add('drop-target');
        return;
    }
    e.dataTransfer.dropEffect = 'none';
    fileListDiv.classList.remove('drop-target');
}

function handleDragLeaveGeneral(e) {
    fileListDiv.classList.remove('drop-target');
}

async function handleDropGeneral(e) {
    e.preventDefault();

    // *** CRITICAL FIX HERE ***
    // Ensure draggedItemId is available and the item exists
    if (!draggedItemId || !files[draggedItemId]) {
        fileListDiv.classList.remove('drop-target'); // Ensure no lingering highlight
        return;
    }
    // *** END CRITICAL FIX ***

    const itemToMove = files[draggedItemId];
    // This `handleDropGeneral` is specifically for dropping into the *root* folder.
    // If you want to drop into the `currentPath` *always* when dragging onto blank space,
    // you would use `currentPath` here instead of `'/'`.
    const targetRootId = '/';

    if (itemToMove.parent === targetRootId) {
        console.log("Item is already in the root folder.");
        fileListDiv.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    // Prevent moving a folder into its own subfolder (if targetRootId is a descendant)
    if (itemToMove.type === 'folder' && isDescendant(targetRootId, draggedItemId)) {
        console.warn("Cannot move a folder into its own subfolder.");
        fileListDiv.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    const success = await performMove(draggedItemId, targetRootId); // Move to root
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
    // Clean up any lingering 'drop-target' classes
    document.querySelectorAll('.drop-target').forEach(el => el.classList.remove('drop-target'));
    draggedItemId = null; // Clear the dragged item ID
}


// --- Event Listeners ---
createFolderBtn.addEventListener('click', createFolder);
createFileBtn.addEventListener('click', () => createFile('note'));
importDownloadBtn.addEventListener('click', openImportDialog);
// goBackBtn.addEventListener('click', goBack); // Less relevant with tree view, but can keep if desired for path bar navigation

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