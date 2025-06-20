// server.js - WebSocket server for screen sharing application
// Run with: node server.js

const WebSocket = require('ws');
const http = require('http');

const server = http.createServer();
const wss = new WebSocket.Server({ server });

// Data structures
const sessions = new Map(); // sessionId -> session info
const channels = new Map(); // channelId -> channel info
const connections = new Map(); // ws -> session info

let sessionIdCounter = 1;
let channelIdCounter = 1;

// Channel structure
class Channel {
    constructor(id, name, password, hostId) {
        this.id = id;
        this.name = name;
        this.password = password;
        this.hostId = hostId;
        this.users = new Map();
        this.isPasswordProtected = !!password;
        this.createdAt = Date.now();
    }
}

// Session structure
class Session {
    constructor(id, username, ws) {
        this.id = id;
        this.username = username;
        this.ws = ws;
        this.channelId = null;
        this.isStreaming = false;
        this.joinedAt = Date.now();
    }
}

// Helper functions
function broadcast(channel, message, excludeSession = null) {
    channel.users.forEach((user, sessionId) => {
        if (sessionId !== excludeSession && user.ws.readyState === WebSocket.OPEN) {
            user.ws.send(JSON.stringify(message));
        }
    });
}

function sendToSession(sessionId, message) {
    const session = sessions.get(sessionId);
    if (session && session.ws.readyState === WebSocket.OPEN) {
        session.ws.send(JSON.stringify(message));
    }
}

function getChannelList() {
    const channelList = [];
    channels.forEach((channel, id) => {
        channelList.push({
            id: id,
            name: channel.name,
            userCount: channel.users.size,
            isPasswordProtected: channel.isPasswordProtected,
            hostId: channel.hostId
        });
    });
    return channelList;
}

function getChannelUsers(channelId) {
    const channel = channels.get(channelId);
    if (!channel) return [];
    
    const users = [];
    channel.users.forEach((user, sessionId) => {
        users.push({
            sessionId: sessionId,
            username: user.username,
            isHost: channel.hostId === sessionId,
            isStreaming: user.isStreaming,
            joinedAt: user.joinedAt
        });
    });
    return users;
}

// WebSocket connection handler
wss.on('connection', (ws) => {
    console.log('New WebSocket connection');
    
    let currentSession = null;
    
    // Handle messages
    ws.on('message', (data) => {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'login':
                    handleLogin(ws, message);
                    break;
                    
                case 'keep_alive':
                    // Just acknowledge
                    break;
                    
                case 'get_channels':
                    handleGetChannels(ws);
                    break;
                    
                case 'create_channel':
                    handleCreateChannel(ws, message);
                    break;
                    
                case 'join_channel':
                    handleJoinChannel(ws, message);
                    break;
                    
                case 'leave_channel':
                    handleLeaveChannel(ws, message);
                    break;
                    
                case 'get_channel_users':
                    handleGetChannelUsers(ws, message);
                    break;
                    
                case 'start_stream':
                    handleStartStream(ws, message);
                    break;
                    
                case 'stop_stream':
                    handleStopStream(ws, message);
                    break;
                    
                case 'offer':
                case 'answer':
                case 'ice_candidate':
                    handleWebRTCSignaling(ws, message);
                    break;
                    
                case 'request_idr':
                    handleRequestIDR(ws, message);
                    break;
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    });
    
    // Handle disconnection
    ws.on('close', () => {
        if (currentSession) {
            handleDisconnect(currentSession);
        }
    });
    
    // Message handlers
    function handleLogin(ws, message) {
        const sessionId = sessionIdCounter++;
        const session = new Session(sessionId, message.username || `User${sessionId}`, ws);
        
        sessions.set(sessionId, session);
        connections.set(ws, session);
        currentSession = session;
        
        ws.send(JSON.stringify({
            type: 'login_response',
            sessionId: sessionId,
            username: session.username
        }));
        
        console.log(`User ${session.username} (${sessionId}) logged in`);
    }
    
    function handleGetChannels(ws) {
        ws.send(JSON.stringify({
            type: 'channel_list',
            channels: getChannelList()
        }));
    }
    
    function handleCreateChannel(ws, message) {
        const session = connections.get(ws);
        if (!session) return;
        
        const channelId = channelIdCounter++;
        const channel = new Channel(
            channelId.toString(),
            message.name || `Channel ${channelId}`,
            message.password,
            session.id
        );
        
        channels.set(channel.id, channel);
        
        ws.send(JSON.stringify({
            type: 'channel_created',
            channelId: channel.id,
            name: channel.name
        }));
        
        console.log(`Channel ${channel.name} (${channel.id}) created by ${session.username}`);
    }
    
    function handleJoinChannel(ws, message) {
        const session = connections.get(ws);
        if (!session) return;
        
        const channel = channels.get(message.channelId);
        if (!channel) {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Channel not found'
            }));
            return;
        }
        
        // Check password
        if (channel.isPasswordProtected && channel.password !== message.password) {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Invalid password'
            }));
            return;
        }
        
        // Leave current channel if any
        if (session.channelId) {
            const oldChannel = channels.get(session.channelId);
            if (oldChannel) {
                oldChannel.users.delete(session.id);
                broadcast(oldChannel, {
                    type: 'user_left',
                    sessionId: session.id,
                    username: session.username
                });
            }
        }
        
        // Join new channel
        session.channelId = channel.id;
        channel.users.set(session.id, session);
        
        ws.send(JSON.stringify({
            type: 'joined_channel',
            channelId: channel.id,
            channelName: channel.name,
            users: getChannelUsers(channel.id)
        }));
        
        // Notify other users
        broadcast(channel, {
            type: 'user_joined',
            sessionId: session.id,
            username: session.username
        }, session.id);
        
        console.log(`${session.username} joined channel ${channel.name}`);
    }
    
    function handleLeaveChannel(ws, message) {
        const session = connections.get(ws);
        if (!session || !session.channelId) return;
        
        const channel = channels.get(session.channelId);
        if (!channel) return;
        
        channel.users.delete(session.id);
        session.channelId = null;
        
        broadcast(channel, {
            type: 'user_left',
            sessionId: session.id,
            username: session.username
        });
        
        // Delete channel if empty and not persistent
        if (channel.users.size === 0) {
            channels.delete(channel.id);
            console.log(`Channel ${channel.name} deleted (empty)`);
        }
        
        ws.send(JSON.stringify({
            type: 'left_channel'
        }));
    }
    
    function handleGetChannelUsers(ws, message) {
        const users = getChannelUsers(message.channelId);
        ws.send(JSON.stringify({
            type: 'channel_users',
            channelId: message.channelId,
            users: users
        }));
    }
    
    function handleStartStream(ws, message) {
        const session = connections.get(ws);
        if (!session || !session.channelId) return;
        
        const channel = channels.get(session.channelId);
        if (!channel) return;
        
        session.isStreaming = true;
        
        // Notify all users in channel
        broadcast(channel, {
            type: 'stream_started',
            sessionId: session.id,
            username: session.username
        }, session.id);
        
        console.log(`${session.username} started streaming in ${channel.name}`);
    }
    
    function handleStopStream(ws, message) {
        const session = connections.get(ws);
        if (!session || !session.channelId) return;
        
        const channel = channels.get(session.channelId);
        if (!channel) return;
        
        session.isStreaming = false;
        
        // Notify all users in channel
        broadcast(channel, {
            type: 'stream_stopped',
            sessionId: session.id,
            username: session.username
        }, session.id);
        
        console.log(`${session.username} stopped streaming`);
    }
    
    function handleWebRTCSignaling(ws, message) {
        const session = connections.get(ws);
        if (!session || !session.channelId) return;
        
        const channel = channels.get(session.channelId);
        if (!channel) return;
        
        // Forward WebRTC signaling to other users in channel
        const forwardMessage = {
            ...message,
            from: session.id
        };
        
        if (message.to) {
            // Send to specific user
            sendToSession(message.to, forwardMessage);
        } else {
            // Broadcast to all users in channel
            broadcast(channel, forwardMessage, session.id);
        }
    }
    
    function handleRequestIDR(ws, message) {
        const session = connections.get(ws);
        if (!session || !session.channelId) return;
        
        const channel = channels.get(message.channelId);
        if (!channel) return;
        
        // Find streaming users and request IDR
        channel.users.forEach((user, sessionId) => {
            if (user.isStreaming && sessionId !== session.id) {
                sendToSession(sessionId, {
                    type: 'request_idr',
                    from: session.id
                });
            }
        });
    }
    
    function handleDisconnect(session) {
        if (session.channelId) {
            const channel = channels.get(session.channelId);
            if (channel) {
                channel.users.delete(session.id);
                broadcast(channel, {
                    type: 'user_left',
                    sessionId: session.id,
                    username: session.username
                });
                
                if (channel.users.size === 0) {
                    channels.delete(channel.id);
                }
            }
        }
        
        sessions.delete(session.id);
        connections.delete(session.ws);
        
        console.log(`User ${session.username} (${session.id}) disconnected`);
    }
});

// Start server
const PORT = process.env.PORT || 43254;
server.listen(PORT, () => {
    console.log(`WebSocket server running on ws://localhost:${PORT}`);
    console.log('Clients can connect using the Screen Share Application');
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, closing server...');
    wss.close(() => {
        server.close(() => {
            console.log('Server closed');
            process.exit(0);
        });
    });
});