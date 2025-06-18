// Node.js WebSocket to TCP/UDP Proxy
// This bridges browser WebSocket connections to your existing C++ TCP/UDP server
// Save as: proxy-server.js
// Run with: node proxy-server.js

const WebSocket = require('ws');
const net = require('net');
const dgram = require('dgram');

class WebSocketTCPUDPProxy {
    constructor(wsPort = 43255, tcpHost = 'localhost', tcpPort = 43254, udpPort = 43254) {
        this.wsPort = wsPort;
        this.tcpHost = tcpHost;
        this.tcpPort = tcpPort;
        this.udpPort = udpPort;
        this.connections = new Map();
        this.sessionToConnection = new Map();
        
        // Create UDP socket for communicating with C++ server
        this.udpSocket = dgram.createSocket('udp4');
        this.udpSocket.bind(0); // Bind to random port
        
        this.setupUDPHandling();
        this.setupWebSocketServer();
        console.log(`WebSocket to TCP/UDP Proxy started`);
        console.log(`WebSocket Server: ws://localhost:${wsPort}`);
        console.log(`TCP Target: ${tcpHost}:${tcpPort}`);
        console.log(`UDP Target: ${tcpHost}:${udpPort}`);
        console.log(`Configure browser to connect to: localhost:${wsPort}`);
    }
    
    setupUDPHandling() {
        this.udpSocket.on('message', (data, rinfo) => {
            try {
                // Parse UDP response from C++ server
                if (data.length >= 14) { // Minimum UDP packet size
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    const sessionId = view.getUint32(0, true);
                    const id = view.getUint32(4, true);
                    const identifySecret = view.getBigUint64(8, true);
                    const command = view.getUint16(16, true);
                    
                    console.log(`UDP from server: Session=${sessionId}, Command=${command}, Size=${data.length}`);
                    
                    // Forward to appropriate WebSocket client
                    const ws = this.sessionToConnection.get(sessionId);
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        // Send as special UDP message to browser
                        const udpMessage = new ArrayBuffer(data.length + 1);
                        const udpView = new Uint8Array(udpMessage);
                        udpView[0] = 0xFF; // UDP marker
                        udpView.set(new Uint8Array(data), 1);
                        
                        ws.send(udpMessage);
                        console.log(`Forwarded UDP to browser: ${data.length} bytes`);
                    }
                }
            } catch (error) {
                console.error('UDP message parsing error:', error);
            }
        });
        
        this.udpSocket.on('error', (error) => {
            console.error('UDP socket error:', error);
        });
    }
    
    setupWebSocketServer() {
        this.wss = new WebSocket.Server({ 
            port: this.wsPort,
            perMessageDeflate: false // Disable compression for binary data
        });
        
        this.wss.on('connection', (ws, req) => {
            console.log(`WebSocket client connected from ${req.socket.remoteAddress}`);
            this.handleWebSocketConnection(ws);
        });
        
        this.wss.on('error', (error) => {
            console.error('WebSocket Server Error:', error);
        });
    }
    
    handleWebSocketConnection(ws) {
        // Create TCP connection to your C++ server
        const tcpSocket = new net.Socket();
        const connectionId = Math.random().toString(36).substr(2, 9);
        let sessionId = null;
        let identifySecret = 0n;
        
        const connection = { ws, tcpSocket, sessionId, identifySecret };
        this.connections.set(connectionId, connection);
        
        // Connect to your C++ server
        tcpSocket.connect(this.tcpPort, this.tcpHost, () => {
            console.log(`[${connectionId}] TCP connection established to ${this.tcpHost}:${this.tcpPort}`);
        });
        
        // WebSocket -> TCP/UDP: Forward browser messages to C++ server
        ws.on('message', (data) => {
            try {
                // Convert Buffer to ArrayBuffer if needed
                let arrayBuffer;
                if (data instanceof ArrayBuffer) {
                    arrayBuffer = data;
                } else if (Buffer.isBuffer(data)) {
                    arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                } else {
                    console.log(`[${connectionId}] Unknown data type:`, typeof data);
                    return;
                }
                
                const dataArray = new Uint8Array(arrayBuffer);
                
                // Check if this is a UDP message (starts with 0xFF marker)
                if (dataArray[0] === 0xFF) {
                    // This is UDP data - remove marker and send via UDP
                    const udpData = dataArray.slice(1);
                    console.log(`[${connectionId}] WebSocket->UDP: ${udpData.length} bytes`);
                    
                    const udpBuffer = Buffer.from(udpData);
                    this.udpSocket.send(udpBuffer, this.udpPort, this.tcpHost, (error) => {
                        if (error) {
                            console.error(`[${connectionId}] UDP send error:`, error);
                        } else {
                            console.log(`[${connectionId}] UDP sent successfully`);
                        }
                    });
                } else {
                    // This is TCP data
                    if (tcpSocket.writable) {
                        console.log(`[${connectionId}] WebSocket->TCP: ${arrayBuffer.byteLength} bytes`);
                        
                        // Parse TCP message to extract session ID for UDP mapping
                        if (arrayBuffer.byteLength >= 6) {
                            const view = new DataView(arrayBuffer);
                            const command = view.getUint16(4, true);
                            console.log(`[${connectionId}] TCP command: ${command}`);
                        }
                        
                        tcpSocket.write(Buffer.from(arrayBuffer));
                    } else {
                        console.log(`[${connectionId}] TCP socket not writable`);
                    }
                }
            } catch (error) {
                console.error(`[${connectionId}] Error forwarding message:`, error);
            }
        });
        
        // TCP -> WebSocket: Forward C++ server responses to browser
        tcpSocket.on('data', (data) => {
            try {
                if (ws.readyState === WebSocket.OPEN) {
                    console.log(`[${connectionId}] TCP->WebSocket: ${data.length} bytes`);
                    
                    // Parse response to extract session ID for UDP mapping
                    if (data.length >= 14 && !sessionId) {
                        try {
                            const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                            const id = view.getUint32(0, true);
                            const command = view.getUint16(4, true);
                            
                            // If this is a login response (command 2), extract session ID
                            if (command === 2 && data.length >= 14) {
                                sessionId = view.getUint32(10, true); // Session ID is at offset 10
                                connection.sessionId = sessionId;
                                this.sessionToConnection.set(sessionId, ws);
                                console.log(`[${connectionId}] Mapped session ${sessionId} to WebSocket`);
                            }
                        } catch (parseError) {
                            // Not a parseable message, continue
                        }
                    }
                    
                    // Convert Buffer to ArrayBuffer for WebSocket
                    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                    ws.send(arrayBuffer);
                } else {
                    console.log(`[${connectionId}] WebSocket not open (state: ${ws.readyState})`);
                }
            } catch (error) {
                console.error(`[${connectionId}] Error forwarding to WebSocket:`, error);
            }
        });
        
        // Handle disconnections
        ws.on('close', (code, reason) => {
            console.log(`[${connectionId}] WebSocket closed: ${code} ${reason}`);
            if (sessionId) {
                this.sessionToConnection.delete(sessionId);
            }
            tcpSocket.destroy();
            this.connections.delete(connectionId);
        });
        
        ws.on('error', (error) => {
            console.error(`[${connectionId}] WebSocket error:`, error);
            if (sessionId) {
                this.sessionToConnection.delete(sessionId);
            }
            tcpSocket.destroy();
            this.connections.delete(connectionId);
        });
        
        tcpSocket.on('close', () => {
            console.log(`[${connectionId}] TCP connection closed`);
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            if (sessionId) {
                this.sessionToConnection.delete(sessionId);
            }
            this.connections.delete(connectionId);
        });
        
        tcpSocket.on('error', (error) => {
            console.error(`[${connectionId}] TCP error:`, error);
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            if (sessionId) {
                this.sessionToConnection.delete(sessionId);
            }
            this.connections.delete(connectionId);
        });
        
        tcpSocket.setTimeout(60000); // 60 seconds
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
}

// Configuration
const WS_PORT = process.argv[2] ? parseInt(process.argv[2]) : 43255;
const TCP_HOST = process.argv[3] || 'localhost';
const TCP_PORT = process.argv[4] ? parseInt(process.argv[4]) : 43254;
const UDP_PORT = process.argv[5] ? parseInt(process.argv[5]) : 43254;

// Start proxy
const proxy = new WebSocketTCPUDPProxy(WS_PORT, TCP_HOST, TCP_PORT, UDP_PORT);

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down proxy...');
    const stats = proxy.getStats();
    console.log(`Closing ${stats.activeConnections} active connections`);
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('Received SIGTERM, shutting down...');
    process.exit(0);
});

// Status logging
setInterval(() => {
    const stats = proxy.getStats();
    if (stats.activeConnections > 0) {
        console.log(`Status: ${stats.activeConnections} connections, ${stats.sessionMappings} session mappings`);
    }
}, 30000);