const suggestedProducts = new Set();

async function sendMessage() {
    const input = document.getElementById('user-input');
    const query = input.value.trim();
    if (!query) return;

    const siteId = new URLSearchParams(window.location.search).get('site_id') || 1;

    // 1. Append User Message
    appendMessage('user', query);
    input.value = '';

    // Show thinking state
    const thinkingId = showThinkingState();

    try {
        const response = await fetch('/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                site_id: parseInt(siteId),
                query: query,
                conversation_id: window.currentConversationId
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Update conversation ID to maintain state
        window.currentConversationId = data.conversation_id;

        // Ensure the answer is just the text, even if the model returned a JSON string
        let botAnswer = data.answer;
        if (typeof botAnswer === 'string' && (botAnswer.trim().startsWith('{') || botAnswer.trim().startsWith('['))) {
            try {
                const parsed = JSON.parse(botAnswer);
                botAnswer = parsed.answer || parsed.text || botAnswer;
            } catch (e) {
                // Not valid JSON, keep as is
            }
        }

        // 2. Append Bot Message
        appendMessage('bot', botAnswer, data.retrieved_products);

    } catch (error) {
        console.error('Error sending message:', error);
        appendMessage('bot', 'Sorry, something went wrong. Please try again later.', []);
    } finally {
        removeThinkingState(thinkingId);
    }
}

function showThinkingState() {
    const chatWindow = document.getElementById('chat-window');
    const id = 'thinking-' + Date.now();

    const messageDiv = document.createElement('div');
    messageDiv.id = id;
    messageDiv.className = 'flex justify-start items-end space-x-2';

    messageDiv.innerHTML = `
        <img src="https://ui-avatars.com/api/?name=E-bot&background=2563eb&color=fff"
             class="w-8 h-8 rounded-full shrink-0 mb-1" alt="Bot Avatar">
        <div class="flex flex-col items-start w-full">
            <div class="message-bubble p-3 rounded-2xl shadow-sm bg-gray-200 text-gray-800 rounded-bl-none">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;

    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return id;
}

function removeThinkingState(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendMessage(role, text, products = []) {
    const chatWindow = document.getElementById('chat-window');
    const now = new Date();
    const timestamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const messageDiv = document.createElement('div');
    messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} items-end space-x-2`;

    if (role === 'bot') {
        messageDiv.innerHTML = `
            <img src="https://ui-avatars.com/api/?name=E-bot&background=2563eb&color=fff"
                 class="w-8 h-8 rounded-full shrink-0 mb-1" alt="Bot Avatar">
        `;
    }

    const bubbleContainer = document.createElement('div');
    bubbleContainer.className = `flex flex-col ${role === 'user' ? 'items-end' : 'items-start'} w-full`;

    const bubble = document.createElement('div');
    bubble.className = `message-bubble p-3 rounded-2xl shadow-sm ${
        role === 'user'
            ? 'bg-blue-600 text-white rounded-br-none'
            : 'bg-gray-200 text-gray-800 rounded-bl-none'
    }`;
    bubble.innerText = text;

    const time = document.createElement('span');
    time.className = 'text-[10px] text-gray-400 mt-1';
    time.innerText = timestamp;

    bubbleContainer.appendChild(bubble);
    bubbleContainer.appendChild(time);

    if (role === 'bot') {
        // Render Products
        if (products && products.length > 0) {
            updateSuggestedProducts(products);

            const productList = document.createElement('div');
            productList.className = 'flex overflow-x-auto space-x-3 py-2 w-full';
            productList.style.scrollbarWidth = 'none';

            products.forEach(p => {
                const card = document.createElement('div');
                card.className = 'bg-white border rounded-lg p-2 shadow-sm shrink-0 w-32 text-xs';
                card.innerHTML = `
                    <div class="font-bold truncate">${p.product_name || 'Unknown Product'}</div>
                    <div class="text-blue-600 font-semibold">${p.price || 'N/A'} ${p.currency || ''}</div>
                `;
                productList.appendChild(card);
            });

            const botWrapper = document.createElement('div');
            botWrapper.className = 'flex flex-col items-start max-w-full';
            botWrapper.appendChild(bubbleContainer);
            botWrapper.appendChild(productList);

            messageDiv.appendChild(botWrapper);
        } else {
            messageDiv.appendChild(bubbleContainer);
        }
    } else {
        messageDiv.appendChild(bubbleContainer);
    }

    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function updateSuggestedProducts(products) {
    if (!products) return;
    products.forEach(p => {
        if (p.product_name) {
            suggestedProducts.add(p.product_name);
        }
    });
    renderProductsDropdown();
}

function renderProductsDropdown() {
    const list = document.getElementById('products-list');
    list.innerHTML = '';

    if (suggestedProducts.size === 0) {
        list.innerHTML = '<div class="p-3 text-xs text-gray-500 italic">No products suggested yet</div>';
        return;
    }

    suggestedProducts.forEach(name => {
        const item = document.createElement('div');
        item.className = 'p-3 text-sm border-b last:border-b-0 hover:bg-gray-100 cursor-pointer transition-colors';
        item.innerText = name;
        list.appendChild(item);
    });
}


document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Dropdown logic
document.getElementById('dropdown-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const dropdown = document.getElementById('products-dropdown');
    dropdown.classList.toggle('hidden');
    if (!dropdown.classList.contains('hidden')) {
        renderProductsDropdown();
    }
});

window.addEventListener('click', () => {
    const dropdown = document.getElementById('products-dropdown');
    if (!dropdown.classList.contains('hidden')) {
        dropdown.classList.add('hidden');
    }
});

window.addEventListener('click', (e) => {
    const dropdown = document.getElementById('products-dropdown');
    if (dropdown.classList.contains('hidden')) return;

    if (!dropdown.contains(e.target) && e.target.id !== 'dropdown-btn') {
        dropdown.classList.add('hidden');
    }
});
