// Service Worker for Quantum Bet PWA
const CACHE_NAME = 'quantum-bet-v1.0.0';
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  // Add more static assets as needed
];

// Install Service Worker
self.addEventListener('install', event => {
  console.log('Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Service Worker: Caching files');
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        console.log('Service Worker: Installed');
        return self.skipWaiting();
      })
  );
});

// Activate Service Worker
self.addEventListener('activate', event => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache');
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker: Activated');
      return self.clients.claim();
    })
  );
});

// Fetch Strategy: Network First with Cache Fallback
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip Chrome extension and other protocols
  if (!event.request.url.startsWith('http')) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Check if response is valid
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        // Clone response for cache
        const responseToCache = response.clone();

        // Cache successful responses
        caches.open(CACHE_NAME)
          .then(cache => {
            cache.put(event.request, responseToCache);
          });

        return response;
      })
      .catch(() => {
        // Network failed, try cache
        return caches.match(event.request)
          .then(response => {
            if (response) {
              console.log('Service Worker: Serving from cache');
              return response;
            }
            
            // If not in cache and offline, show offline page
            if (event.request.destination === 'document') {
              return new Response(
                `
                <!DOCTYPE html>
                <html>
                <head>
                  <title>Quantum Bet - Offline</title>
                  <meta name="viewport" content="width=device-width, initial-scale=1">
                  <style>
                    body {
                      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                      display: flex;
                      flex-direction: column;
                      align-items: center;
                      justify-content: center;
                      height: 100vh;
                      margin: 0;
                      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      color: white;
                      text-align: center;
                      padding: 20px;
                      box-sizing: border-box;
                    }
                    .offline-icon {
                      font-size: 4rem;
                      margin-bottom: 1rem;
                    }
                    .offline-title {
                      font-size: 2rem;
                      margin-bottom: 0.5rem;
                    }
                    .offline-message {
                      font-size: 1.1rem;
                      opacity: 0.9;
                      margin-bottom: 2rem;
                      max-width: 400px;
                    }
                    .retry-button {
                      background: rgba(255,255,255,0.2);
                      border: 2px solid white;
                      color: white;
                      padding: 0.8rem 2rem;
                      border-radius: 25px;
                      font-size: 1rem;
                      cursor: pointer;
                      transition: all 0.3s ease;
                    }
                    .retry-button:hover {
                      background: white;
                      color: #667eea;
                    }
                  </style>
                </head>
                <body>
                  <div class="offline-icon">⚽</div>
                  <h1 class="offline-title">Quantum Bet</h1>
                  <p class="offline-message">
                    You're currently offline. Please check your connection and try again.
                  </p>
                  <button class="retry-button" onclick="window.location.reload()">
                    Try Again
                  </button>
                  
                  <script>
                    // Auto-retry when online
                    window.addEventListener('online', () => {
                      window.location.reload();
                    });
                  </script>
                </body>
                </html>
                `,
                {
                  headers: { 'Content-Type': 'text/html' }
                }
              );
            }
          });
      })
  );
});

// Background Sync for offline actions
self.addEventListener('sync', event => {
  console.log('Service Worker: Background sync triggered');
  
  if (event.tag === 'background-sync') {
    event.waitUntil(
      // Handle offline actions when back online
      handleBackgroundSync()
    );
  }
});

// Push Notifications
self.addEventListener('push', event => {
  console.log('Service Worker: Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'New betting opportunity available!',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    tag: 'quantum-bet-notification',
    requireInteraction: true,
    actions: [
      {
        action: 'view',
        title: 'View Details',
        icon: '/static/icons/view-icon.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/static/icons/close-icon.png'
      }
    ],
    data: {
      url: '/?notification=true'
    }
  };

  event.waitUntil(
    self.registration.showNotification('Quantum Bet', options)
  );
});

// Notification Click Handler
self.addEventListener('notificationclick', event => {
  console.log('Service Worker: Notification clicked');
  
  event.notification.close();

  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/')
    );
  }
});

// Handle Background Sync
async function handleBackgroundSync() {
  try {
    // Sync offline actions, update cache, etc.
    console.log('Service Worker: Handling background sync');
    
    // Example: Sync offline betting actions
    const offlineActions = await getOfflineActions();
    
    for (const action of offlineActions) {
      try {
        await syncAction(action);
        await removeOfflineAction(action.id);
      } catch (error) {
        console.error('Service Worker: Failed to sync action', error);
      }
    }
    
  } catch (error) {
    console.error('Service Worker: Background sync failed', error);
  }
}

// Helper functions for IndexedDB operations
async function getOfflineActions() {
  // Return offline actions from IndexedDB
  return [];
}

async function syncAction(action) {
  // Sync individual action with server
  return fetch('/api/sync', {
    method: 'POST',
    body: JSON.stringify(action),
    headers: {
      'Content-Type': 'application/json'
    }
  });
}

async function removeOfflineAction(actionId) {
  // Remove synced action from IndexedDB
  console.log('Removing offline action:', actionId);
}