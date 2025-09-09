// Main JavaScript for Crypto Portfolio AI Dashboard

// Global variables
let currentPage = 'dashboard';
let refreshIntervals = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set up page-specific functionality
    const path = window.location.pathname;
    
    if (path === '/coins') {
        currentPage = 'coins';
        initializeCoinsPage();
    } else if (path === '/news') {
        currentPage = 'news';
        initializeNewsPage();
    } else if (path === '/whale') {
        currentPage = 'whale';
        initializeWhalePage();
    } else {
        currentPage = 'dashboard';
        initializeDashboard();
    }
    
    // Set up global event listeners
    setupGlobalEventListeners();
    
    // Start auto-refresh for real-time data
    startAutoRefresh();
}

function setupGlobalEventListeners() {
    // Handle navigation clicks
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', function(e) {
            // Add loading state
            showLoading();
        });
    });
    
    // Handle window resize for responsive charts
    window.addEventListener('resize', function() {
        // Redraw charts if they exist
        if (typeof Chart !== 'undefined') {
            Chart.helpers.each(Chart.instances, function(chart) {
                chart.resize();
            });
        }
    });
}

function initializeDashboard() {
    // Load dashboard-specific data
    loadDashboardStats();
    
    // Set up auto-refresh for dashboard
    refreshIntervals.dashboard = setInterval(loadDashboardStats, 30000); // 30 seconds
}

function initializeCoinsPage() {
    // Load crypto data
    loadCryptoData();
    
    // Set up auto-refresh for crypto prices
    refreshIntervals.coins = setInterval(loadCryptoData, 60000); // 1 minute
}

function initializeNewsPage() {
    // Load news data
    loadNewsData();
    
    // Set up auto-refresh for news
    refreshIntervals.news = setInterval(loadNewsData, 120000); // 2 minutes
}

function initializeWhalePage() {
    // Load whale detection data
    loadWhaleData();
    
    // Set up auto-refresh for whale detection
    refreshIntervals.whale = setInterval(loadWhaleData, 30000); // 30 seconds
}

// Utility functions
function showLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.classList.remove('hidden');
    }
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.classList.add('hidden');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500 text-white' :
        type === 'error' ? 'bg-red-500 text-white' :
        type === 'warning' ? 'bg-yellow-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${
                type === 'success' ? 'fa-check-circle' :
                type === 'error' ? 'fa-exclamation-circle' :
                type === 'warning' ? 'fa-exclamation-triangle' :
                'fa-info-circle'
            } mr-2"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(amount);
}

function formatPercentage(value, decimals = 2) {
    return `${value.toFixed(decimals)}%`;
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        return Math.floor(diff / 60000) + ' minutes ago';
    } else if (diff < 86400000) { // Less than 1 day
        return Math.floor(diff / 3600000) + ' hours ago';
    } else {
        return Math.floor(diff / 86400000) + ' days ago';
    }
}

function getSentimentColor(sentiment) {
    switch(sentiment.toLowerCase()) {
        case 'positive':
        case 'bullish':
            return 'text-green-600';
        case 'negative':
        case 'bearish':
            return 'text-red-600';
        default:
            return 'text-gray-600';
    }
}

function getSentimentIcon(sentiment) {
    switch(sentiment.toLowerCase()) {
        case 'positive':
        case 'bullish':
            return 'fa-thumbs-up';
        case 'negative':
        case 'bearish':
            return 'fa-thumbs-down';
        default:
            return 'fa-minus';
    }
}

// API helper functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showNotification('Failed to load data. Please try again.', 'error');
        throw error;
    }
}

// Data loading functions
async function loadDashboardStats() {
    try {
        const [portfolioData, whaleData] = await Promise.all([
            apiCall('/api/portfolio/status'),
            apiCall('/api/whale/detect')
        ]);
        
        // Update portfolio value
        const portfolioElement = document.getElementById('portfolio-value');
        if (portfolioElement) {
            portfolioElement.textContent = formatCurrency(portfolioData.balance);
        }
        
        // Update whale events count
        const whaleElement = document.getElementById('whale-events');
        if (whaleElement) {
            whaleElement.textContent = whaleData.detection_summary.total_events;
        }
        
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
    }
}

async function loadCryptoData() {
    try {
        const data = await apiCall('/api/crypto/prices');
        
        // Update crypto grid if on coins page
        if (currentPage === 'coins' && typeof displayCryptoGrid === 'function') {
            displayCryptoGrid(data);
        }
        
    } catch (error) {
        console.error('Error loading crypto data:', error);
    }
}

async function loadNewsData() {
    try {
        const data = await apiCall('/api/news');
        
        // Update news display if on news page
        if (currentPage === 'news' && typeof displayNewsData === 'function') {
            displayNewsData(data);
        }
        
    } catch (error) {
        console.error('Error loading news data:', error);
    }
}

async function loadWhaleData() {
    try {
        const data = await apiCall('/api/whale/detect');
        
        // Update whale display if on whale page
        if (currentPage === 'whale' && typeof displayWhaleEvents === 'function') {
            displayWhaleEvents(data);
        }
        
    } catch (error) {
        console.error('Error loading whale data:', error);
    }
}

// Auto-refresh management
function startAutoRefresh() {
    // Clear any existing intervals
    Object.values(refreshIntervals).forEach(interval => clearInterval(interval));
    refreshIntervals = {};
    
    // Start page-specific auto-refresh
    if (currentPage === 'dashboard') {
        refreshIntervals.dashboard = setInterval(loadDashboardStats, 30000);
    } else if (currentPage === 'coins') {
        refreshIntervals.coins = setInterval(loadCryptoData, 60000);
    } else if (currentPage === 'news') {
        refreshIntervals.news = setInterval(loadNewsData, 120000);
    } else if (currentPage === 'whale') {
        refreshIntervals.whale = setInterval(loadWhaleData, 30000);
    }
}

function stopAutoRefresh() {
    Object.values(refreshIntervals).forEach(interval => clearInterval(interval));
    refreshIntervals = {};
}

// Chart utilities
function createChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, config);
}

function updateChart(chart, newData) {
    if (!chart) return;
    
    chart.data = newData;
    chart.update();
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'error');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showNotification('A network error occurred. Please check your connection.', 'error');
});

// Export functions for use in other scripts
window.CryptoDashboard = {
    showLoading,
    hideLoading,
    showNotification,
    formatCurrency,
    formatPercentage,
    formatTimestamp,
    getSentimentColor,
    getSentimentIcon,
    apiCall,
    createChart,
    updateChart
};
