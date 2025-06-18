// Simple Pass-Through WebSocket to TCP/UDP Proxy - NO MESSAGE PARSING
// Save as: fixed-proxy-server.js
// Run with: node fixed-proxy-server.js [ws_port] [tcp_host] [tcp_port]

const WebSocket = require('ws');
const net = require('net');
const dgram = require('dgram');

class FixedWebSocketTCPUDPProxy {
    constructor(wsPort = 43255, tcpHost = 'localhost', tcpPort = 43254, udpPort = 43254) {
        this.wsPort = wsPort;
        this.tcpHost = tcpHost;
        this.tcpPort = tcpPort;
        this.udpPort = udpPort;
        this.udpReceivePort = udpPort + 1; // Use next port for receiving stream data
        this.connections = new Map();
        this.sessionToConnection = new Map();
        
        // Create UDP socket for communicating with C++ server
        this.udpSocket = dgram.createSocket('udp4');
        
        this.setupUDPHandling();
        this.setupWebSocketServer();
        
        console.log(`✅ FIXED WebSocket to TCP/UDP Proxy started`);
        console.log(`📡 WebSocket Server: ws://localhost:${wsPort}`);
        console.log(`🔗 TCP Target: ${tcpHost}:${tcpPort}`);
        console.log(`📡 UDP Target: ${tcpHost}:${udpPort} (outbound)`);
        console.log(`📡 UDP Receive: ${tcpHost}:${this.udpReceivePort} (inbound stream data)`);
        console.log(`🔧 FIXED: Simple pass-through, NO message parsing`);
    }
    
    setupUDPHandling() {
        this.udpSocket.on('error', (error) => {
            console.error('❌ UDP socket error:', error);
        });
        
        // Bind to receive port to get stream data from C++ server
        try {
            this.udpSocket.bind(this.udpReceivePort, () => {
                console.log(`📡 UDP socket bound to port ${this.udpReceivePort} for receiving stream data`);
                console.log(`💡 Configure your C++ server to send UDP stream data to port ${this.udpReceivePort}`);
            });
        } catch (error) {
            console.error(`❌ Failed to bind UDP to port ${this.udpReceivePort}:`, error);
            // Fall back to random port
            this.udpSocket.bind(0, () => {
                const address = this.udpSocket.address();
                console.log(`📡 UDP socket bound to random port ${address.port}`);
                console.log(`💡 Configure your C++ server to send UDP stream data to port ${address.port}`);
            });
        }
        
        // Listen for UDP stream data FROM the C++ server
        this.udpSocket.on('message', (data, rinfo) => {
            console.log(`📡 Received UDP stream data: ${data.length} bytes from ${rinfo.address}:${rinfo.port}`);
            
            // For now, forward to ALL connected WebSocket clients
            // Later we can parse session ID if needed
            let forwarded = 0;
            for (const [connectionId, connection] of this.connections) {
                if (connection.ws.readyState === WebSocket.OPEN && connection.sessionId) {
                    try {
                        // Add 0xFF marker and forward stream data
                        const markedData = Buffer.concat([Buffer.from([0xFF]), data]);
                        const arrayBuffer = markedData.buffer.slice(markedData.byteOffset, markedData.byteOffset + markedData.byteLength);
                        connection.ws.send(arrayBuffer);
                        forwarded++;
                    } catch (error) {
                        console.error(`❌ Error forwarding to ${connectionId}:`, error);
                    }
                }
            }
            console.log(`📤 Forwarded stream data to ${forwarded} WebSocket clients`);
        });
        
        console.log(`📡 UDP socket ready for C++ server communication`);
    }
    
    setupWebSocketServer() {
        this.wss = new WebSocket.Server({ 
            port: this.wsPort,
            perMessageDeflate: false
        });
        
        this.wss.on('connection', (ws, req) => {
            const clientIP = req.socket.remoteAddress;
            console.log(`🔌 WebSocket client connected from ${clientIP}`);
            this.handleWebSocketConnection(ws);
        });
        
        this.wss.on('error', (error) => {
            console.error('❌ WebSocket Server Error:', error);
        });
    }
    
    handleWebSocketConnection(ws) {
        const tcpSocket = new net.Socket();
        const connectionId = Math.random().toString(36).substr(2, 9);
        let sessionId = null;
        
        const connection = { ws, tcpSocket, sessionId, lastActivity: Date.now() };
        this.connections.set(connectionId, connection);
        
        console.log(`🔗 [${connectionId}] Connecting to ${this.tcpHost}:${this.tcpPort}`);
        
        tcpSocket.connect(this.tcpPort, this.tcpHost, () => {
            console.log(`✅ [${connectionId}] TCP connected`);
        });
        
        // WebSocket -> TCP: Simple pass-through
        ws.on('message', (data) => {
            try {
                connection.lastActivity = Date.now();
                
                let buffer;
                if (Buffer.isBuffer(data)) {
                    buffer = data;
                } else if (data instanceof ArrayBuffer) {
                    buffer = Buffer.from(data);
                } else {
                    console.log(`❌ [${connectionId}] Unknown data type`);
                    return;
                }
                
                // Check for UDP message (starts with 0xFF)
                if (buffer[0] === 0xFF) {
                    const udpData = buffer.slice(1);
                    console.log(`📡 [${connectionId}] WebSocket->UDP: ${udpData.length} bytes`);
                    
                    this.udpSocket.send(udpData, this.udpPort, this.tcpHost, (error) => {
                        if (error) {
                            console.error(`❌ [${connectionId}] UDP send error:`, error);
                        }
                    });
                } else {
                    // TCP message - SIMPLE PASS-THROUGH
                    if (tcpSocket.writable) {
                        console.log(`📤 [${connectionId}] WebSocket->TCP: ${buffer.length} bytes`);
                        
                        // MINIMAL SESSION EXTRACTION - only for login responses
                        if (buffer.length >= 6) {
                            const command = buffer.readUInt16LE(4);
                            if (command === 1) { // Login request
                                console.log(`🔐 [${connectionId}] Login request detected`);
                            }
                        }
                        
                        tcpSocket.write(buffer);
                    }
                }
            } catch (error) {
                console.error(`❌ [${connectionId}] Error forwarding message:`, error);
            }
        });
        
        // TCP -> WebSocket: COMPLETE PASS-THROUGH - NO PARSING
        tcpSocket.on('data', (data) => {
            try {
                connection.lastActivity = Date.now();
                
                if (ws.readyState !== WebSocket.OPEN) {
                    console.log(`⚠️ [${connectionId}] WebSocket not open`);
                    return;
                }
                
                console.log(`📥 [${connectionId}] TCP->WebSocket: ${data.length} bytes (direct pass-through)`);
                
                // MINIMAL SESSION EXTRACTION - only look for login responses
                if (data.length >= 14) {
                    try {
                        const id = data.readUInt32LE(0);
                        const command = data.readUInt16LE(4);
                        
                        if (command === 2) { // Response_Login
                            const accountId = data.readUInt32LE(6);
                            const extractedSessionId = data.readUInt32LE(10);
                            connection.sessionId = extractedSessionId;
                            this.sessionToConnection.set(extractedSessionId, ws);
                            console.log(`🔐 [${connectionId}] Session ${extractedSessionId} mapped (Account: ${accountId})`);
                        }
                    } catch (e) {
                        // Ignore parsing errors - just pass through
                    }
                }
                
                // DIRECT PASS-THROUGH - Convert Buffer to ArrayBuffer
                const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                ws.send(arrayBuffer);
                
            } catch (error) {
                console.error(`❌ [${connectionId}] Error forwarding to WebSocket:`, error);
            }
        });
        
        // Handle disconnections with cleanup
        ws.on('close', (code, reason) => {
            console.log(`🔌 [${connectionId}] WebSocket closed: ${code}`);
            this.cleanup(connectionId);
        });
        
        ws.on('error', (error) => {
            console.error(`❌ [${connectionId}] WebSocket error:`, error);
            this.cleanup(connectionId);
        });
        
        tcpSocket.on('close', () => {
            console.log(`🔗 [${connectionId}] TCP connection closed`);
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            this.cleanup(connectionId);
        });
        
        tcpSocket.on('error', (error) => {
            console.error(`❌ [${connectionId}] TCP error:`, error);
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            this.cleanup(connectionId);
        });
        
        // SIMPLE timeout handling
        tcpSocket.setTimeout(120000); // 2 minutes
        tcpSocket.on('timeout', () => {
            console.log(`⏰ [${connectionId}] TCP socket timeout`);
            tcpSocket.destroy();
        });
        
        // SIMPLE keepalive - only send if we have a session
        const keepAliveInterval = setInterval(() => {
            if (connection.sessionId && ws.readyState === WebSocket.OPEN && tcpSocket.writable) {
                // Send simple TCP keepalive (command 0)
                const keepAlive = Buffer.alloc(6);
                keepAlive.writeUInt32LE(999999, 0); // Unique ID for keepalives
                keepAlive.writeUInt16LE(0, 4); // KeepAlive command
                tcpSocket.write(keepAlive);
                console.log(`💓 [${connectionId}] Sent keepalive`);
            }
        }, 30000); // Every 30 seconds
        
        // Clean up interval on disconnect
        tcpSocket.on('close', () => {
            clearInterval(keepAliveInterval);
        });
        ws.on('close', () => {
            clearInterval(keepAliveInterval);
        });
    }
    
    cleanup(connectionId) {
        const connection = this.connections.get(connectionId);
        if (connection && connection.sessionId) {
            this.sessionToConnection.delete(connection.sessionId);
        }
        this.connections.delete(connectionId);
    }
    
    getStats() {
        return {
            activeConnections: this.connections.size,
            sessionMappings: this.sessionToConnection.size,
            wsPort: this.wsPort,
            tcpTarget: `${this.tcpHost}:${this.tcpPort}`,
            udpTarget: `${this.tcpHost}:${this.udpPort}`
        };
    }
    
    logStats() {
        const stats = this.getStats();
        if (stats.activeConnections > 0) {
            console.log(`📊 Status: ${stats.activeConnections} connections, ${stats.sessionMappings} sessions mapped`);
        }
    }
}

// Configuration
const WS_PORT = process.argv[2] ? parseInt(process.argv[2]) : 43255;
const TCP_HOST = process.argv[3] || 'localhost';
const TCP_PORT = process.argv[4] ? parseInt(process.argv[4]) : 43254;
const UDP_PORT = process.argv[5] ? parseInt(process.argv[5]) : 43254;

// Validate ports
if (isNaN(WS_PORT) || WS_PORT < 1 || WS_PORT > 65535) {
    console.error('❌ Invalid WebSocket port');
    process.exit(1);
}

if (isNaN(TCP_PORT) || TCP_PORT < 1 || TCP_PORT > 65535) {
    console.error('❌ Invalid TCP port');
    process.exit(1);
}

console.log(`🚀 Starting SIMPLE Proxy Server...`);
console.log(`⚙️ Config: WS=${WS_PORT}, TCP=${TCP_HOST}:${TCP_PORT}, UDP=${TCP_HOST}:${UDP_PORT}`);

// Start simple proxy
const proxy = new FixedWebSocketTCPUDPProxy(WS_PORT, TCP_HOST, TCP_PORT, UDP_PORT);

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down proxy...');
    const stats = proxy.getStats();
    console.log(`📊 Final stats: ${stats.activeConnections} connections`);
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('🛑 Received SIGTERM, shutting down...');
    process.exit(0);
});

// Status logging every 30 seconds
setInterval(() => {
    proxy.logStats();
}, 30000);

console.log(`✅ SIMPLE proxy server running!`);
console.log(`🌐 Connect your browser to: ws://localhost:${WS_PORT}`);

module.exports = FixedWebSocketTCPUDPProxy;