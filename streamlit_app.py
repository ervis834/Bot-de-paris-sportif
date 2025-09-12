mport React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Menu, Home, TrendingUp, DollarSign, Settings, Bot, Plus, X, ChevronDown, Check, AlertCircle, Trophy, Target, Zap, Shield, Brain, Activity } from 'lucide-react';

// Configuration API
const API_CONFIG = {
  OPENAI_API_KEY: process.env.REACT_APP_OPENAI_API_KEY || '',
  BACKEND_URL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'
};

// Hook pour la connexion WebSocket temps réel
const useWebSocket = (url) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };
    
    setSocket(ws);
    
    return () => {
      ws.close();
    };
  }, [url]);
  
  const sendMessage = (message) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message));
    }
  };
  
  return { isConnected, lastMessage, sendMessage };
};

// Composant principal de l'application
const QuantumBetApp = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: "Bonjour! Je suis votre assistant IA Quantum Max avec GPT-5. Comment puis-je vous aider aujourd'hui?",
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [portfolio, setPortfolio] = useState({
    bankroll: 12547,
    roi: 0.123,
    winRate: 0.678,
    dailyPnl: 247
  });
  const [selectedBets, setSelectedBets] = useState([]);
  const [showBetSlip, setShowBetSlip] = useState(false);
  
  const chatEndRef = useRef(null);
  const { isConnected, lastMessage, sendMessage: sendWsMessage } = useWebSocket(API_CONFIG.WS_URL);
  
  // Scroll automatique vers le bas du chat
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Traitement des messages WebSocket
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'prediction_update') {
        setPredictions(lastMessage.data);
      } else if (lastMessage.type === 'portfolio_update') {
        setPortfolio(lastMessage.data);
      }
    }
  }, [lastMessage]);
  
  // Fonction pour appeler GPT-5 (simulation avec fallback GPT-4)
  const callGPT5 = async (prompt) => {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_CONFIG.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-4-turbo-preview', // Utilisera GPT-5 quand disponible
          messages: [
            {
              role: 'system',
              content: `Tu es l'assistant IA de Bot Quantum Max, un système avancé de prédiction sportive et de paris intelligents.
              
              CAPACITÉS:
              - Créer des combinés optimisés
              - Analyser les matchs en temps réel
              - Gérer le portefeuille de paris
              - Trouver des value bets
              - Prédire avec des modèles ML avancés
              
              STYLE:
              - Professionnel mais accessible
              - Précis dans les probabilités
              - Proactif dans les recommandations
              
              COMMANDES SPÉCIALES:
              - /combo [critères] : Créer un combiné
              - /predict [match] : Prédiction détaillée
              - /portfolio : Statut du portefeuille
              - /value : Chercher des value bets
              - /live : Mises à jour en temps réel`
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.3,
          max_tokens: 500
        })
      });
      
      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      console.error('Erreur GPT-5:', error);
      return "Désolé, une erreur s'est produite. Veuillez réessayer.";
    }
  };
  
  // Traitement des commandes du chatbot
  const processCommand = async (message) => {
    const lowerMessage = message.toLowerCase();
    
    // Détection des intentions
    if (lowerMessage.includes('combiné') || lowerMessage.includes('combo')) {
      return await createCombo(message);
    } else if (lowerMessage.includes('prédiction') || lowerMessage.includes('predict')) {
      return await getPredictions(message);
    } else if (lowerMessage.includes('portefeuille') || lowerMessage.includes('portfolio')) {
      return getPortfolioStatus();
    } else if (lowerMessage.includes('value') || lowerMessage.includes('valeur')) {
      return await findValueBets(message);
    } else {
      // Utiliser GPT-5 pour les requêtes complexes
      return await callGPT5(message);
    }
  };
  
  // Création de combinés
  const createCombo = async (message) => {
    // Parser les critères depuis le message
    const matches = message.match(/(\d+)\s*match/i);
    const odds = message.match(/cote[s]?\s*[><=]+\s*([\d.]+)/i);
    
    const numMatches = matches ? parseInt(matches[1]) : 3;
    const minOdds = odds ? parseFloat(odds[1]) : 1.5;
    
    // Appel API backend
    try {
      const response = await fetch(`${API_CONFIG.BACKEND_URL}/api/create-combo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_matches: numMatches,
          min_odds: minOdds,
          type: 'balanced'
        })
      });
      
      const combo = await response.json();
      
      // Formatter la réponse
      return {
        type: 'combo',
        data: combo,
        message: `✅ Combiné créé avec succès!
        
**${combo.selections.length} sélections**
- Cote totale: **${combo.total_odds.toFixed(2)}**
- Mise recommandée: **${(combo.recommended_stake * 100).toFixed(1)}%** du bankroll
- Valeur attendue: **${(combo.expected_value * 100).toFixed(1)}%**

${combo.selections.map(s => `• ${s.match}: ${s.selection} @${s.odds.toFixed(2)}`).join('\n')}`
      };
    } catch (error) {
      return "Erreur lors de la création du combiné. Veuillez réessayer.";
    }
  };
  
  // Récupération des prédictions
  const getPredictions = async (message) => {
    try {
      const response = await fetch(`${API_CONFIG.BACKEND_URL}/api/predictions`);
      const data = await response.json();
      
      setPredictions(data.predictions);
      
      return {
        type: 'predictions',
        data: data.predictions,
        message: `🔮 **${data.predictions.length} prédictions disponibles**
        
Top 3 avec la meilleure confiance:
${data.predictions.slice(0, 3).map(p => 
  `• ${p.home_team} vs ${p.away_team}
   Prédiction: **${p.predicted_outcome}**
   Confiance: **${(p.confidence * 100).toFixed(1)}%**`
).join('\n\n')}`
      };
    } catch (error) {
      return "Erreur lors de la récupération des prédictions.";
    }
  };
  
  // Statut du portefeuille
  const getPortfolioStatus = () => {
    return {
      type: 'portfolio',
      data: portfolio,
      message: `💰 **Statut du Portefeuille**
      
• Bankroll: **€${portfolio.bankroll.toLocaleString()}**
• ROI: **${(portfolio.roi * 100).toFixed(1)}%**
• Win Rate: **${(portfolio.winRate * 100).toFixed(1)}%**
• P&L Aujourd'hui: **€${portfolio.dailyPnl > 0 ? '+' : ''}${portfolio.dailyPnl}**

Performance: ${portfolio.roi > 0.1 ? '🟢 Excellente' : portfolio.roi > 0 ? '🟡 Positive' : '🔴 À surveiller'}`
    };
  };
  
  // Recherche de value bets
  const findValueBets = async (message) => {
    try {
      const response = await fetch(`${API_CONFIG.BACKEND_URL}/api/value-bets`);
      const data = await response.json();
      
      return {
        type: 'value_bets',
        data: data.bets,
        message: `💎 **${data.bets.length} Value Bets trouvés**
        
${data.bets.slice(0, 5).map(b => 
  `• ${b.match}
   ${b.selection} - Cote: **${b.odds}**
   Edge: **${(b.edge * 100).toFixed(1)}%**
   Confiance: **${(b.confidence * 100).toFixed(1)}%**`
).join('\n\n')}`
      };
    } catch (error) {
      return "Erreur lors de la recherche de value bets.";
    }
  };
  
  // Envoi de message
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);
    
    // Traiter la commande
    const response = await processCommand(inputMessage);
    
    setIsTyping(false);
    
    const botMessage = {
      type: 'bot',
      content: typeof response === 'string' ? response : response.message,
      data: response.data,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, botMessage]);
    
    // Envoyer via WebSocket pour synchronisation
    sendWsMessage({
      type: 'chat_message',
      content: inputMessage,
      response: response
    });
  };
  
  // Actions rapides
  const quickActions = [
    { label: "Combiné du jour", command: "Crée un combiné avec 3 matchs d'aujourd'hui ayant des cotes > 1.5" },
    { label: "Top prédictions", command: "Montre-moi les 5 meilleures prédictions" },
    { label: "Value bets", command: "Trouve des value bets avec un edge > 5%" },
    { label: "Statut portfolio", command: "Quel est le statut de mon portefeuille?" }
  ];
  
  // Composant de carte de prédiction
  const PredictionCard = ({ prediction }) => {
    const confidenceColor = prediction.confidence > 0.7 ? 'text-green-500' : 
                           prediction.confidence > 0.5 ? 'text-yellow-500' : 'text-red-500';
    
    return (
      <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 mb-3">
        <div className="flex justify-between items-start mb-2">
          <div>
            <h4 className="font-semibold text-gray-900">
              {prediction.home_team} vs {prediction.away_team}
            </h4>
            <p className="text-sm text-gray-500">{prediction.league} • {prediction.time}</p>
          </div>
          <span className={`text-sm font-bold ${confidenceColor}`}>
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="flex justify-between items-center mt-3">
          <span className="text-sm font-medium text-indigo-600">
            {prediction.predicted_outcome}
          </span>
          <button
            onClick={() => setSelectedBets(prev => [...prev, prediction])}
            className="px-3 py-1 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Ajouter
          </button>
        </div>
      </div>
    );
  };
  
  // Interface principale
  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-indigo-50 to-purple-50">
      {/* Header avec statut de connexion */}
      <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-4 shadow-lg">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8" />
            <div>
              <h1 className="text-xl font-bold">Bot Quantum Max</h1>
              <p className="text-xs opacity-90">GPT-5 Powered</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
            <span className="text-sm">{isConnected ? 'En ligne' : 'Hors ligne'}</span>
          </div>
        </div>
      </header>
      
      {/* Contenu principal basé sur l'onglet actif */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'home' && (
          <div className="h-full p-4 overflow-y-auto">
            {/* Statistiques rapides */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <Trophy className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">ROI</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">
                  {(portfolio.roi * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Win Rate</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">
                  {(portfolio.winRate * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <DollarSign className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Bankroll</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">
                  €{portfolio.bankroll.toLocaleString()}
                </p>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-purple-500" />
                  <span className="text-xs text-gray-500">P&L Jour</span>
                </div>
                <p className={`text-2xl font-bold ${portfolio.dailyPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  €{portfolio.dailyPnl > 0 ? '+' : ''}{portfolio.dailyPnl}
                </p>
              </div>
            </div>
            
            {/* Actions rapides */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Actions Rapides</h3>
              <div className="grid grid-cols-2 gap-2">
                {quickActions.map((action, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setActiveTab('chat');
                      setInputMessage(action.command);
                    }}
                    className="bg-white border border-indigo-200 text-indigo-600 p-3 rounded-xl text-sm font-medium hover:bg-indigo-50 transition-colors flex items-center justify-center space-x-2"
                  >
                    <Zap className="w-4 h-4" />
                    <span>{action.label}</span>
                  </button>
                ))}
              </div>
            </div>
            
            {/* Prédictions du jour */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Top Prédictions</h3>
              {predictions.slice(0, 3).map((pred, idx) => (
                <PredictionCard key={idx} prediction={pred} />
              ))}
            </div>
          </div>
        )}
        
        {activeTab === 'chat' && (
          <div className="h-full flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${
                    msg.type === 'user' 
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white' 
                      : 'bg-white border border-gray-200'
                  } rounded-2xl p-4 shadow-sm`}>
                    {msg.type === 'bot' && (
                      <div className="flex items-center space-x-2 mb-2">
                        <Bot className="w-5 h-5 text-indigo-600" />
                        <span className="text-sm font-semibold text-gray-700">Quantum AI</span>
                      </div>
                    )}
                    <div className={`${msg.type === 'bot' ? 'text-gray-800' : ''} whitespace-pre-wrap`}>
                      {msg.content}
                    </div>
                    {msg.data && msg.data.type === 'combo' && (
                      <button
                        onClick={() => {
                          setSelectedBets(msg.data.selections);
                          setShowBetSlip(true);
                        }}
                        className="mt-3 w-full bg-indigo-100 text-indigo-700 py-2 rounded-lg font-medium hover:bg-indigo-200 transition-colors"
                      >
                        Voir le ticket
                      </button>
                    )}
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={chatEndRef} />
            </div>
            
            {/* Zone de saisie */}
            <div className="border-t border-gray-200 bg-white p-4">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Tapez votre commande..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim()}
                  className="px-4 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
              
              {/* Suggestions de commandes */}
              <div className="mt-3 flex flex-wrap gap-2">
                {quickActions.slice(0, 2).map((action, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInputMessage(action.command)}
                    className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-200 transition-colors"
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'predictions' && (
          <div className="h-full p-4 overflow-y-auto">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Prédictions</h2>
            
            {/* Filtres */}
            <div className="bg-white rounded-xl p-3 mb-4 shadow-sm">
              <div className="flex items-center justify-between">
                <select className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                  <option>Toutes les ligues</option>
                  <option>Premier League</option>
                  <option>La Liga</option>
                  <option>Ligue 1</option>
                </select>
                
                <select className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                  <option>Confiance > 60%</option>
                  <option>Confiance > 70%</option>
                  <option>Confiance > 80%</option>
                </select>
              </div>
            </div>
            
            {/* Liste des prédictions */}
            <div className="space-y-3">
              {predictions.map((pred, idx) => (
                <PredictionCard key={idx} prediction={pred} />
              ))}
            </div>
          </div>
        )}
        
        {activeTab === 'portfolio' && (
          <div className="h-full p-4 overflow-y-auto">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Portfolio</h2>
            
            {/* Graphique de performance */}
            <div className="bg-white rounded-xl p-4 mb-4 shadow-sm">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Performance (30j)</h3>
              <div className="h-32 bg-gradient-to-t from-indigo-50 to-transparent rounded-lg flex items-end justify-around px-2">
                {[...Array(15)].map((_, i) => (
                  <div
                    key={i}
                    className="w-4 bg-indigo-500 rounded-t"
                    style={{ height: `${Math.random() * 100}%` }}
                  />
                ))}
              </div>
            </div>
            
            {/* Métriques détaillées */}
            <div className="space-y-3">
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Bankroll Total</span>
                  <span className="font-bold text-gray-900">€{portfolio.bankroll.toLocaleString()}</span>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">ROI</span>
                  <span className={`font-bold ${portfolio.roi >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {(portfolio.roi * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Taux de Réussite</span>
                  <span className="font-bold text-gray-900">{(portfolio.winRate * 100).toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Sharpe Ratio</span>
                  <span className="font-bold text-gray-900">1.45</span>
                </div>
              </div>
            </div>
            
            {/* Bouton de gestion avancée */}
            <button
              onClick={() => setActiveTab('chat')}
              className="w-full mt-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 rounded-xl font-medium hover:opacity-90 transition-opacity"
            >
              Optimiser le Portfolio avec l'IA
            </button>
          </div>
        )}
        
        {activeTab === 'settings' && (
          <div className="h-full p-4 overflow-y-auto">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Paramètres</h2>
            
            <div className="space-y-4">
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <h3 className="font-semibold text-gray-900 mb-3">Modèle IA</h3>
                <select className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                  <option>GPT-5 Turbo (Recommandé)</option>
                  <option>GPT-4 Turbo</option>
                  <option>Claude 3 Opus</option>
                </select>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <h3 className="font-semibold text-gray-900 mb-3">Notifications</h3>
                <div className="space-y-2">
                  <label className="flex items-center justify-between">
                    <span className="text-gray-700">Prédictions haute confiance</span>
                    <input type="checkbox" defaultChecked className="toggle" />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-700">Opportunités d'arbitrage</span>
                    <input type="checkbox" defaultChecked className="toggle" />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-700">Alertes portfolio</span>
                    <input type="checkbox" defaultChecked className="toggle" />
                  </label>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <h3 className="font-semibold text-gray-900 mb-3">Gestion des Risques</h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm text-gray-600">Seuil de confiance minimum</label>
                    <input type="range" min="50" max="95" defaultValue="70" className="w-full" />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>50%</span>
                      <span>70%</span>
                      <span>95%</span>
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-sm text-gray-600">Kelly Fraction Max</label>
                    <input type="range" min="10" max="50" defaultValue="25" className="w-full" />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>10%</span>
                      <span>25%</span>
                      <span>50%</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <h3 className="font-semibold text-gray-900 mb-3">API & Connexions</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">OpenAI GPT-5</span>
                    <span className={`text-xs px-2 py-1 rounded-full ${API_CONFIG.OPENAI_API_KEY ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                      {API_CONFIG.OPENAI_API_KEY ? 'Connecté' : 'Non configuré'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">WebSocket</span>
                    <span className={`text-xs px-2 py-1 rounded-full ${isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                      {isConnected ? 'Connecté' : 'Déconnecté'}
                    </span>
                  </div>
                </div>
              </div>
              
              <button className="w-full bg-red-500 text-white py-3 rounded-xl font-medium hover:bg-red-600 transition-colors">
                Réinitialiser les Paramètres
              </button>
            </div>
          </div>
        )}
      </div>
      
      {/* Bet Slip flottant */}
      {showBetSlip && selectedBets.length > 0 && (
        <div className="absolute bottom-20 left-4 right-4 bg-white rounded-2xl shadow-2xl p-4 max-h-96 overflow-y-auto">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-bold text-gray-900">Ticket de Paris ({selectedBets.length})</h3>
            <button onClick={() => setShowBetSlip(false)}>
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
          
          <div className="space-y-2 mb-3">
            {selectedBets.map((bet, idx) => (
              <div key={idx} className="bg-gray-50 rounded-lg p-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="font-medium">{bet.match || `${bet.home_team} vs ${bet.away_team}`}</span>
                  <button onClick={() => setSelectedBets(prev => prev.filter((_, i) => i !== idx))}>
                    <X className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>{bet.selection || bet.predicted_outcome}</span>
                  <span>@{bet.odds || '1.75'}</span>
                </div>
              </div>
            ))}
          </div>
          
          <div className="border-t pt-3">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-600">Cote totale:</span>
              <span className="font-bold">
                {selectedBets.reduce((acc, bet) => acc * (bet.odds || 1.75), 1).toFixed(2)}
              </span>
            </div>
            
            <div className="flex space-x-2">
              <input
                type="number"
                placeholder="Mise (€)"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <button className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium hover:opacity-90 transition-opacity">
                Valider
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Barre de navigation inférieure */}
      <nav className="bg-white border-t border-gray-200 px-4 py-2">
        <div className="flex justify-around">
          <button
            onClick={() => setActiveTab('home')}
            className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors ${
              activeTab === 'home' ? 'text-indigo-600 bg-indigo-50' : 'text-gray-600'
            }`}
          >
            <Home className="w-5 h-5" />
            <span className="text-xs mt-1">Accueil</span>
          </button>
          
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors relative ${
              activeTab === 'chat' ? 'text-indigo-600 bg-indigo-50' : 'text-gray-600'
            }`}
          >
            <Bot className="w-5 h-5" />
            <span className="text-xs mt-1">IA Chat</span>
            {messages.length > 1 && (
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            )}
          </button>
          
          <button
            onClick={() => setActiveTab('predictions')}
            className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors ${
              activeTab === 'predictions' ? 'text-indigo-600 bg-indigo-50' : 'text-gray-600'
            }`}
          >
            <TrendingUp className="w-5 h-5" />
            <span className="text-xs mt-1">Prédictions</span>
          </button>
          
          <button
            onClick={() => setActiveTab('portfolio')}
            className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors ${
              activeTab === 'portfolio' ? 'text-indigo-600 bg-indigo-50' : 'text-gray-600'
            }`}
          >
            <DollarSign className="w-5 h-5" />
            <span className="text-xs mt-1">Portfolio</span>
          </button>
          
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors ${
              activeTab === 'settings' ? 'text-indigo-600 bg-indigo-50' : 'text-gray-600'
            }`}
          >
            <Settings className="w-5 h-5" />
            <span className="text-xs mt-1">Réglages</span>
          </button>
        </div>
      </nav>
      
      {/* Indicateur de mise à jour en temps réel */}
      {isConnected && lastMessage && (
        <div className="absolute top-16 left-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg animate-slideDown">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4" />
            <span className="text-sm">Mise à jour en temps réel</span>
          </div>
        </div>
      )}
    </div>
  );
};

// CSS pour les animations et styles personnalisés
const styles = `
  @keyframes slideDown {
    from {
      transform: translateY(-100%);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }
  
  .animate-slideDown {
    animation: slideDown 0.3s ease-out;
  }
  
  .toggle {
    appearance: none;
    width: 44px;
    height: 24px;
    background: #cbd5e0;
    border-radius: 9999px;
    position: relative;
    transition: background 0.3s;
    cursor: pointer;
  }
  
  .toggle:checked {
    background: linear-gradient(to right, #6366f1, #a855f7);
  }
  
  .toggle::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s;
  }
  
  .toggle:checked::after {
    transform: translateX(20px);
  }
  
  /* Effet de glissement pour mobile */
  .swipe-indicator {
    position: fixed;
    bottom: 70px;
    left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 4px;
    background: rgba(0,0,0,0.2);
    border-radius: 2px;
  }
  
  /* Animation de pulsation */
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
  
  /* Optimisations PWA */
  * {
    -webkit-tap-highlight-color: transparent;
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    user-select: none;
  }
  
  input, textarea {
    -webkit-user-select: text;
    user-select: text;
  }
  
  /* Scrollbar personnalisée */
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 10px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #555;
  }
`;

// Injection des styles
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement("style");
  styleSheet.innerText = styles;
  document.head.appendChild(styleSheet);
}

export default QuantumBetApp;
