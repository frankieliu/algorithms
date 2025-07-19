document.addEventListener('DOMContentLoaded', initializeExplorer);

const FILE_STORAGE_KEY = 'virtualFileExplorerFiles';
const EXPANDED_FOLDERS_KEY = 'virtualFileExplorerExpandedFolders';

let currentPath = '/';
let files = {};
let expandedFolders = {};
let selectedItemId = null;
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
    const result = await chrome.storage.local.get([FILE_STORAGE_KEY, EXPANDED_FOLDERS_KEY]);
    files = result[FILE_STORAGE_KEY] || {
        '/': { id: '/', name: 'Root', type: 'folder', parent: null, children: [] }
    };
    expandedFolders = result[EXPANDED_FOLDERS_KEY] || { '/': true };

    if (!files['/']) {
        files['/'] = { id: '/', name: 'Root', type: 'folder', parent: null, children: [] };
    }
}

async function saveExpandedFolders() {
    await chrome.storage.local.set({ [EXPANDED_FOLDERS_KEY]: expandedFolders });
}

function renderHierarchy(parentId, level, targetElement) {
    const children = files[parentId].children
        .map(id => files[id])
        .filter(Boolean)
        .sort((a, b) => {
            if (a.type === 'folder' && b.type !== 'folder') return -1;
            if (a.type !== 'folder' && b.type === 'folder') return 1;
            return a.name.localeCompare(b.name);
        });

    const indentationPx = level * 15;

    children.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.classList.add(item.type === 'folder' ? 'folder-item' : 'file-item');
        itemDiv.style.setProperty('--indentation-level', `${indentationPx}px`);
        if (selectedItemId === item.id) {
            itemDiv.classList.add('selected');
        }

        // --- Start of DRAG RELATED ATTRIBUTES/LISTENERS ---
        // Ensure draggable is set consistently for all items that can be dragged
        itemDiv.setAttribute('draggable', 'true'); // Explicitly 'true' string value
        itemDiv.dataset.id = item.id; // Ensure data-id is set for draggedItemId retrieval
        itemDiv.dataset.type = item.type; // Ensure data-type is set for checks

        itemDiv.addEventListener('dragstart', handleDragStart);
        itemDiv.addEventListener('dragend', handleDragEnd);

        // Only folders are drop targets for other items
        if (item.type === 'folder') {
            itemDiv.addEventListener('dragover', handleDragOverItem);
            itemDiv.addEventListener('dragleave', handleDragLeaveItem);
            itemDiv.addEventListener('drop', handleDropItem);
        }
        // --- End of DRAG RELATED ATTRIBUTES/LISTENERS ---

        if (item.type === 'folder') {
            const toggleSpan = document.createElement('span');
            toggleSpan.classList.add('folder-toggle');
            toggleSpan.textContent = expandedFolders[item.id] ? '▼' : '▶';
            toggleSpan.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleFolder(item.id);
            });
            itemDiv.appendChild(toggleSpan);
        } else {
            const emptyToggleSpace = document.createElement('span');
            emptyToggleSpace.classList.add('folder-toggle');
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


        itemDiv.addEventListener('click', (e) => {
            if (!e.defaultPrevented) {
                selectItem(item.id);
            }
        });

        targetElement.appendChild(itemDiv);

        if (item.type === 'folder' && expandedFolders[item.id]) {
            renderHierarchy(item.id, level + 1, targetElement);
        }
    });
}

async function renderFileList() {
    fileListDiv.innerHTML = '';
    currentPathSpan.textContent = getFullPath(currentPath);

    if (files['/']) {
        renderHierarchy('/', 0, fileListDiv);
    } else {
        fileListDiv.textContent = 'No root folder found! This is an error.';
    }

    // Attach general drag-over/leave/drop handlers to the main fileListDiv
    // These should always be present to allow dropping into the root or current path.
    fileListDiv.removeEventListener('dragover', handleDragOverGeneral);
    fileListDiv.removeEventListener('dragleave', handleDragLeaveGeneral);
    fileListDiv.removeEventListener('drop', handleDropGeneral);

    fileListDiv.addEventListener('dragover', handleDragOverGeneral);
    fileListDiv.addEventListener('dragleave', handleDragLeaveGeneral);
    fileListDiv.addEventListener('drop', handleDropGeneral);

    if (selectedItemId && !files[selectedItemId]) {
        hideItemDetails();
    }
}

async function toggleFolder(folderId) {
    expandedFolders[folderId] = !expandedFolders[folderId];
    await saveExpandedFolders();
    renderFileList();
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
            content: content
        };
        files[currentPath].children.push(newFileId);
        await saveFilesToStorage();
        renderFileList();
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

    function deleteRecursive(itemIdToDelete) {
        const item = files[itemIdToDelete];
        if (!item) return;

        if (item.type === 'folder' && item.children) {
            [...item.children].forEach(childId => deleteRecursive(childId));
        }
        delete files[itemIdToDelete];
        delete expandedFolders[itemIdToDelete];
    }

    const parent = files[itemToDelete.parent];
    if (parent && parent.children) {
        parent.children = parent.children.filter(id => id !== itemToDelete.id);
    }

    deleteRecursive(selectedItemId);

    selectedItemId = null;
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
            displayItemDetails(selectedItemId);
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
    moveFolderSelect.innerHTML = '';

    const itemToMove = files[selectedItemId];

    const allFolders = Object.values(files).filter(item => item.type === 'folder');

    allFolders.forEach(folder => {
        if (folder.id === selectedItemId || isDescendant(folder.id, selectedItemId)) {
            return;
        }
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
    while (current && current.parent !== null) {
        segments.unshift(current.name);
        current = files[current.parent];
    }
    if (current && current.id === '/') {
        segments.unshift(current.name);
    }
    return segments;
}

function getFullPath(itemId) {
    if (!files[itemId]) return '';
    const pathSegments = getPathSegments(itemId);
    if (itemId === '/') {
        return '/';
    }
    return '/' + pathSegments.join('/');
}

async function confirmMoveItem() {
    if (!selectedItemId) return;

    const destinationFolderId = moveFolderSelect.value;
    if (!destinationFolderId || confirmMoveBtn.disabled) {
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

async function performMove(itemId, destinationFolderId) {
    const itemToMove = files[itemId];
    const oldParent = files[itemToMove.parent];
    const newParent = files[destinationFolderId];

    if (!itemToMove || !oldParent || !newParent || newParent.type !== 'folder') {
        console.error('Move failed: Invalid item or folders.', {itemId, destinationFolderId, itemToMove, oldParent, newParent});
        return false;
    }

    if (itemToMove.parent === destinationFolderId) {
        console.log('Item is already in the target folder. No move needed.');
        return true;
    }

    if (itemToMove.type === 'folder' && isDescendant(destinationFolderId, itemId)) {
        alert("Cannot move a folder into its own subfolder.");
        return false;
    }

    oldParent.children = oldParent.children.filter(id => id !== itemId);
    console.log(`Removed ${itemToMove.name} from old parent ${oldParent.name}. New children:`, oldParent.children);

    newParent.children.push(itemId);
    itemToMove.parent = destinationFolderId;
    console.log(`Added ${itemToMove.name} to new parent ${newParent.name}. New children:`, newParent.children);

    await saveFilesToStorage();
    if (!expandedFolders[destinationFolderId]) {
        expandedFolders[destinationFolderId] = true;
        await saveExpandedFolders();
    }

    return true;
}

// --- Downloads Integration ---

async function openImportDialog() {
    downloadSelect.innerHTML = '';
    const downloads = await chrome.downloads.search({});

    if (downloads.length === 0) {
        const option = document.createElement('option');
        option.textContent = 'No downloads found.';
        option.disabled = true;
        downloadSelect.appendChild(option);
        confirmImportBtn.disabled = true;
    } else {
        downloads.forEach(dl => {
            if (dl.filename) {
                const option = document.createElement('option');
                option.value = dl.id;
                option.textContent = dl.filename.split(/[\\/]/).pop();
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
        const newFileId = generateUniqueId();
        files[newFileId] = {
            id: newFileId,
            name: download.filename.split(/[\\/]/).pop(),
            type: 'download',
            parent: currentPath,
            downloadId: download.id,
            downloadPath: download.filename
        };
        files[currentPath].children.push(newFileId);
        await saveFilesToStorage();
        renderFileList();
        importDialog.style.display = 'none';
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

function handleDragStart(e) {
    draggedItemId = e.target.dataset.id;
    e.dataTransfer.setData('text/plain', draggedItemId);
    e.dataTransfer.effectAllowed = 'move';
    e.target.classList.add('dragging');
    console.log('Drag started for:', files[draggedItemId]?.name, 'ID:', draggedItemId);
}

function handleDragOverItem(e) {
    e.preventDefault();

    const targetElement = e.target.closest('.folder-item');
    if (!targetElement) {
        e.dataTransfer.dropEffect = 'none';
        console.log('Drag Over Item: Not over a folder-item. dropEffect = none');
        return;
    }
    const targetFolderId = targetElement.dataset.id;

    if (!draggedItemId || !files[draggedItemId]) {
        e.dataTransfer.dropEffect = 'none';
        targetElement.classList.remove('drop-target');
        console.log('Drag Over Item: No valid dragged item. dropEffect = none');
        return;
    }

    const itemToMove = files[draggedItemId];

    if (targetFolderId === draggedItemId) {
        e.dataTransfer.dropEffect = 'none';
        targetElement.classList.remove('drop-target');
        console.log('Drag Over Item: Dropping onto self. dropEffect = none');
        return;
    }

    if (itemToMove.parent === targetFolderId) {
         e.dataTransfer.dropEffect = 'none';
         targetElement.classList.remove('drop-target');
         console.log('Drag Over Item: Dropping onto current parent. dropEffect = none');
         return;
    }

    if (itemToMove.type === 'folder' && isDescendant(targetFolderId, draggedItemId)) {
        e.dataTransfer.dropEffect = 'none';
        targetElement.classList.remove('drop-target');
        console.log('Drag Over Item: Moving folder into its own subfolder. dropEffect = none');
        return;
    }

    e.dataTransfer.dropEffect = 'move';
    targetElement.classList.add('drop-target');
    console.log(`Drag Over Item: Valid drop target: ${files[targetFolderId]?.name}. dropEffect = move`);
}

function handleDragLeaveItem(e) {
    const targetElement = e.target.closest('.folder-item');
    if (targetElement) {
        targetElement.classList.remove('drop-target');
    }
}

async function handleDropItem(e) {
    e.preventDefault();

    const targetElement = e.target.closest('.folder-item');
    const targetFolderId = targetElement ? targetElement.dataset.id : null;

    if (!draggedItemId || !files[draggedItemId] || !targetFolderId || !files[targetFolderId]) {
        console.warn('Drop Item: Invalid dragged or target item/folder.', { draggedItemId, targetFolderId });
        if (targetElement) targetElement.classList.remove('drop-target');
        return;
    }

    const itemToMove = files[draggedItemId];

    if (targetFolderId === draggedItemId ||
        itemToMove.parent === targetFolderId ||
        (itemToMove.type === 'folder' && isDescendant(targetFolderId, draggedItemId))) {
        console.warn('Drop Item: Invalid drop target detected during drop event (pre-check failed).');
        if (targetElement) targetElement.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    console.log(`Drop Item: Attempting to move item ID ${draggedItemId} to folder ID ${targetFolderId}`);

    const success = await performMove(draggedItemId, targetFolderId);
    if (success) {
        hideItemDetails();
        renderFileList();
    }
    if (targetElement) targetElement.classList.remove('drop-target');
    draggedItemId = null;
}

function handleDragOverGeneral(e) {
    e.preventDefault();

    if (!draggedItemId || !files[draggedItemId]) {
        e.dataTransfer.dropEffect = 'none';
        fileListDiv.classList.remove('drop-target');
        console.log('Drag Over General: No valid dragged item. dropEffect = none');
        return;
    }

    const itemToMove = files[draggedItemId];
    const targetRootId = '/';

    if (itemToMove.parent !== targetRootId && !(itemToMove.type === 'folder' && isDescendant(targetRootId, draggedItemId))) {
        e.dataTransfer.dropEffect = 'move';
        fileListDiv.classList.add('drop-target');
        console.log(`Drag Over General: Valid drop target: Root. dropEffect = move (Item: ${itemToMove.name})`);
        return;
    }
    e.dataTransfer.dropEffect = 'none';
    fileListDiv.classList.remove('drop-target');
    console.log(`Drag Over General: Invalid drop on Root. dropEffect = none (Item: ${itemToMove.name}, Parent: ${files[itemToMove.parent]?.name})`);
}

function handleDragLeaveGeneral(e) {
    fileListDiv.classList.remove('drop-target');
}

async function handleDropGeneral(e) {
    e.preventDefault();

    if (!draggedItemId || !files[draggedItemId]) {
        fileListDiv.classList.remove('drop-target');
        return;
    }

    const itemToMove = files[draggedItemId];
    const targetRootId = '/';

    if (itemToMove.parent === targetRootId) {
        console.log("Drop General: Item is already in the root folder.");
        fileListDiv.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    if (itemToMove.type === 'folder' && isDescendant(targetRootId, draggedItemId)) {
        console.warn("Drop General: Cannot move a folder into its own subfolder.");
        fileListDiv.classList.remove('drop-target');
        draggedItemId = null;
        return;
    }

    console.log(`Drop General: Attempting to move item ID ${draggedItemId} to root folder (${targetRootId})`);

    const success = await performMove(draggedItemId, targetRootId);
    if (success) {
        hideItemDetails();
        renderFileList();
    }
    fileListDiv.classList.remove('drop-target');
    draggedItemId = null;
}

function handleDragEnd(e) {
    const draggedElement = document.querySelector('.dragging');
    if (draggedElement) {
        draggedElement.classList.remove('dragging');
    }
    document.querySelectorAll('.drop-target').forEach(el => el.classList.remove('drop-target'));
    draggedItemId = null;
    console.log('Drag ended.');
}


// --- Event Listeners ---
createFolderBtn.addEventListener('click', createFolder);
createFileBtn.addEventListener('click', () => createFile('note'));
importDownloadBtn.addEventListener('click', openImportDialog);

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
    console.log('Explorer initialized.');
}