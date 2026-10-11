/* ===================================================================
   CUPCAKE WONDERLAND - LÓGICA DEL PANEL DE CONTROL & SIMULADOR
   =================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSimulator();
    loadOrders();
    loadCatalog();
    loadLogs();
    checkHealth();
});

// -------------------------------------------------------------
// NAVEGACIÓN POR PESTAÑAS
// -------------------------------------------------------------
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-item");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            document.querySelectorAll(".tab-pane").forEach(pane => {
                pane.classList.remove("active");
            });

            const activePane = document.getElementById(`tab-${targetTab}`);
            if (activePane) {
                activePane.classList.add("active");
                if (targetTab === "orders") loadOrders();
                if (targetTab === "catalog") loadCatalog();
                if (targetTab === "logs") loadLogs();
            }
        });
    });
}

// -------------------------------------------------------------
// SIMULADOR DE WHATSAPP
// -------------------------------------------------------------
function initSimulator() {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const simPhoneInput = document.getElementById("sim-phone");
    const btnResetChat = document.getElementById("btn-reset-chat");
    const chips = document.querySelectorAll(".chip");

    // Enviar mensaje del formulario
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = "";
        appendMessage("user", text);

        await sendSimulatedMessage(text);
    });

    // Chips de sugerencia rápida
    chips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const msg = chip.getAttribute("data-msg");
            appendMessage("user", msg);
            await sendSimulatedMessage(msg);
        });
    });

    // Reiniciar conversación
    btnResetChat.addEventListener("click", async () => {
        const phone = simPhoneInput.value.trim() || "525512345678";
        try {
            await fetch(`/api/conversations/${phone}`, { method: "DELETE" });
            chatMessages.innerHTML = `
                <div class="message bot">
                    <div class="bubble">
                        ¡Memoria conversacional reiniciada! 🧁✨ ¿En qué podemos ayudarte hoy?
                    </div>
                    <span class="time">${getTimeString()}</span>
                </div>
            `;
            document.getElementById("last-exec-info").innerText = "Memoria borrada con éxito.";
        } catch (err) {
            console.error("Error al reiniciar:", err);
        }
    });

    async function sendSimulatedMessage(messageText) {
        const phone = simPhoneInput.value.trim() || "525512345678";
        const execInfo = document.getElementById("last-exec-info");
        execInfo.innerHTML = "⏳ <i>El agente está procesando...</i>";

        try {
            const res = await fetch("/api/simulate-message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    phone_number: phone,
                    message: messageText
                })
            });

            const data = await res.json();
            appendMessage("bot", data.bot_response);

            execInfo.innerHTML = `
                <b>Modelo:</b> ${data.model_used}<br>
                <b>Latencia:</b> ${data.execution_time_ms} ms<br>
                <b>Teléfono:</b> ${data.phone_number}
            `;

            // Actualizar órdenes en segundo plano por si se creó un pedido nuevo
            loadOrders();
        } catch (err) {
            appendMessage("bot", "⚠️ Error al conectar con el servidor backend.");
            execInfo.innerText = "Error de conexión HTTP.";
        }
    }

    function appendMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}`;
        
        // Reemplazar saltos de línea y formato asteriscos
        const formatted = formatWhatsAppText(text);

        msgDiv.innerHTML = `
            <div class="bubble">${formatted}</div>
            <span class="time">${getTimeString()}</span>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function formatWhatsAppText(text) {
    if (!text) return "";
    return text
        .replace(/\*(.*?)\*/g, '<b>$1</b>')
        .replace(/_(.*?)_/g, '<i>$1</i>');
}

function getTimeString() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// -------------------------------------------------------------
// GESTIÓN DE PEDIDOS
// -------------------------------------------------------------
let allOrders = [];

async function loadOrders() {
    const container = document.getElementById("orders-container");
    const countBadge = document.getElementById("order-count");
    
    try {
        const res = await fetch("/api/orders");
        allOrders = await res.json();

        countBadge.innerText = allOrders.length;
        renderOrders(allOrders);
    } catch (err) {
        container.innerHTML = `<p style="color:#f87171;">Error al cargar pedidos.</p>`;
    }
}

function renderOrders(orders) {
    const container = document.getElementById("orders-container");
    if (!orders || orders.length === 0) {
        container.innerHTML = `<p style="color:var(--text-muted);">No hay pedidos en esta categoría.</p>`;
        return;
    }

    container.innerHTML = orders.map(o => `
        <div class="order-card">
            <div class="order-header">
                <div>
                    <div class="order-id">${o.id}</div>
                    <div class="order-client">👤 ${o.customer_name || 'Cliente'} (${o.phone_number})</div>
                </div>
                <span class="status-badge status-${o.status}">${o.status}</span>
            </div>

            <div class="order-details">
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Cantidad:</span>
                    <b>${o.quantity} cupcakes</b>
                </div>
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Sabor:</span>
                    <span>${o.flavor}</span>
                </div>
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Relleno:</span>
                    <span>${o.filling || 'Sin relleno'}</span>
                </div>
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Cobertura:</span>
                    <span>${o.frosting || 'Buttercream'}</span>
                </div>
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Fecha Entrega:</span>
                    <b style="color:var(--accent-amber);">${o.delivery_date}</b>
                </div>
                <div class="order-details-row">
                    <span style="color:var(--text-muted);">Modalidad:</span>
                    <span>${o.delivery_type === 'domicilio' ? '🛵 Domicilio' : '🏪 Recoger'}</span>
                </div>
            </div>

            <div class="order-footer">
                <div class="order-price">$${o.total_price.toFixed(2)} MXN</div>
                <select class="form-control" style="width: auto; padding: 4px 8px; font-size: 12px;" onchange="updateOrderStatus('${o.id}', this.value)">
                    <option value="COTIZACION" ${o.status === 'COTIZACION' ? 'selected' : ''}>Cotización</option>
                    <option value="PENDIENTE_PAGO" ${o.status === 'PENDIENTE_PAGO' ? 'selected' : ''}>Pendiente Pago</option>
                    <option value="CONFIRMADO" ${o.status === 'CONFIRMADO' ? 'selected' : ''}>Confirmado</option>
                    <option value="EN_PREPARACION" ${o.status === 'EN_PREPARACION' ? 'selected' : ''}>En Preparación</option>
                    <option value="LISTO" ${o.status === 'LISTO' ? 'selected' : ''}>Listo para Entrega</option>
                    <option value="ENTREGADO" ${o.status === 'ENTREGADO' ? 'selected' : ''}>Entregado</option>
                    <option value="CANCELADO" ${o.status === 'CANCELADO' ? 'selected' : ''}>Cancelado</option>
                </select>
            </div>
        </div>
    `).join("");
}

// Filtros de pedidos
document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const filter = btn.getAttribute("data-filter");
        if (filter === "ALL") {
            renderOrders(allOrders);
        } else {
            renderOrders(allOrders.filter(o => o.status === filter));
        }
    });
});

document.getElementById("btn-refresh-orders")?.addEventListener("click", loadOrders);

async function updateOrderStatus(orderId, newStatus) {
    try {
        await fetch(`/api/orders/${orderId}/status`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: newStatus })
        });
        loadOrders();
    } catch (err) {
        alert("Error al actualizar estado del pedido.");
    }
}

// -------------------------------------------------------------
// CATÁLOGO
// -------------------------------------------------------------
async function loadCatalog() {
    try {
        const res = await fetch("/api/catalog");
        const data = await res.json();

        // Sabores
        document.getElementById("catalog-sabores").innerHTML = data.sabores_masa.map(s => `
            <div class="catalog-card">
                <h3>🍰 ${s.nombre}</h3>
                <p>${s.descripcion}</p>
            </div>
        `).join("");

        // Rellenos
        document.getElementById("catalog-rellenos").innerHTML = data.rellenos.map(r => `
            <div class="catalog-card">
                <h3>🍓 ${r.nombre}</h3>
                <p>${r.precio_extra === 0 ? 'Sin costo adicional' : '+$' + r.precio_extra}</p>
            </div>
        `).join("");

        // Coberturas
        document.getElementById("catalog-coberturas").innerHTML = data.coberturas.map(c => `
            <div class="catalog-card">
                <h3>🍦 ${c.nombre}</h3>
                <p>${c.descripcion}</p>
            </div>
        `).join("");

        // Paquetes
        document.getElementById("catalog-paquetes").innerHTML = Object.values(data.paquetes_precios).map(p => `
            <div class="catalog-card">
                <h3>📦 ${p.nombre}</h3>
                <p>${p.descripcion}</p>
                <div class="price">$${p.precio.toFixed(0)} MXN</div>
            </div>
        `).join("");

    } catch (err) {
        console.error("Error al cargar catálogo:", err);
    }
}

// -------------------------------------------------------------
// BITÁCORA Y LOGS
// -------------------------------------------------------------
async function loadLogs() {
    const tbody = document.getElementById("logs-tbody");
    try {
        const res = await fetch("/api/logs");
        const data = await res.json();
        
        if (!data.recent_messages || data.recent_messages.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No hay mensajes registrados aún.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.recent_messages.map(m => `
            <tr>
                <td style="color:var(--text-muted); font-size:12px;">${m.timestamp}</td>
                <td>
                    <span style="font-weight:600; color:${m.direction === 'IN' ? '#38bdf8' : '#34d399'};">
                        ${m.direction === 'IN' ? '📥 ENTRANTE' : '📤 SALIENTE'}
                    </span>
                </td>
                <td>${m.phone_number}</td>
                <td>${escapeHtml(m.message_text)}</td>
                <td><span style="font-size:11px; color:#94a3b8;">${m.status || 'ok'}</span></td>
            </tr>
        `).join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="5" style="color:#f87171;">Error al consultar bitácora.</td></tr>`;
    }
}

document.getElementById("btn-refresh-logs")?.addEventListener("click", loadLogs);

// -------------------------------------------------------------
// HEALTH CHECK
// -------------------------------------------------------------
async function checkHealth() {
    const badgeText = document.getElementById("webhook-status-text");
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.status === "healthy") {
            const configStr = data.whatsapp_configured ? "Meta Conectado" : "Modo Simulación";
            badgeText.innerText = `Webhook Listo (${configStr})`;
        }
    } catch (e) {
        badgeText.innerText = "Servidor desconectado";
    }
}

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
