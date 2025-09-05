"""Examples and templates for the AI chatbot."""

CHATBOT_EXAMPLES = {
    "combo_creation": [
        "Crée un combiné avec 3 matchs d'aujourd'hui ayant des cotes supérieures à 1.50",
        "Génère un combiné conservateur avec des matchs de Premier League",
        "Fais-moi un combiné agressif avec 5 sélections de ce weekend",
        "Crée un système Yankees avec des matchs de Ligue 1",
        "Combiné équilibré avec des cotes entre 1.40 et 2.50"
    ],
    
    "predictions": [
        "Montre-moi les prédictions les plus confiantes pour demain",
        "Quelles sont les meilleures prédictions pour la Premier League ?",
        "Donne-moi les prédictions avec plus de 70% de confiance",
        "Prédictions pour les matchs de Champions League cette semaine",
        "Top 5 des prédictions pour les paris Over/Under"
    ],
    
    "match_analysis": [
        "Analyse le match PSG vs Lyon en détail",
        "Que penses-tu du prochain match de Manchester United ?",
        "Analyse tactique pour Liverpool vs Arsenal",
        "Évalue les chances de victoire du Real Madrid ce weekend",
        "Quel est ton pronostic pour le derby de Manchester ?"
    ],
    
    "portfolio_management": [
        "Quel est le statut de mon portefeuille ?",
        "Montre-moi mes performances cette semaine",
        "Comment va mon ROI ce mois-ci ?",
        "Analyse les risques de mon portefeuille actuel",
        "Recommandations pour optimiser ma bankroll"
    ],
    
    "value_betting": [
        "Trouve-moi des paris à valeur élevée pour aujourd'hui",
        "Quelles sont les meilleures opportunités de value bet ?",
        "Recherche des paris avec un edge supérieur à 5%",
        "Paris sous-cotés pour ce weekend",
        "Opportunités d'arbitrage disponibles"
    ],
    
    "risk_management": [
        "Évalue le risque de parier 100€ sur ce match",
        "Quelle mise Kelly pour ce pari ?",
        "Analyse les corrélations de mes paris actuels",
        "Recommandations pour réduire ma variance",
        "Calcule le Value at Risk de mon portefeuille"
    ],
    
    "alerts_scheduling": [
        "Alerte-moi quand une prédiction dépasse 80% de confiance",
        "Notification si mon ROI tombe sous -5%",
        "Préviens-moi des opportunités d'arbitrage",
        "Alerte pour les matchs de mon équipe favorite",
        "Rappel quotidien du statut de mon portefeuille"
    ],
    
    "data_queries": [
        "Historique des performances sur la Premier League",
        "Statistiques de mes paris sur les derniers 30 jours",
        "Quelles ligues sont les plus rentables pour moi ?",
        "Analyse de corrélation entre météo et résultats",
        "Performance comparative des différents modèles"
    ]
}

COMMAND_TEMPLATES = {
    "create_combo": {
        "description": "Créer un pari combiné selon des critères spécifiques",
        "parameters": {
            "num_matches": "Nombre de matchs (2-8)",
            "min_odds": "Cote minimum par sélection",
            "max_odds": "Cote maximum par sélection",
            "leagues": "Ligues spécifiques (PL, FL1, BL1, SA, PD)",
            "date": "Période (today, tomorrow, week)",
            "type": "Type de combiné (conservative, balanced, aggressive)"
        },
        "example": "Crée un combiné balanced avec 4 matchs de Premier League ayant des cotes entre 1.6 et 2.8"
    },
    
    "get_predictions": {
        "description": "Obtenir des prédictions filtrées selon des critères",
        "parameters": {
            "min_confidence": "Confiance minimum (0.0-1.0)",
            "leagues": "Ligues à inclure",
            "date": "Période des matchs",
            "limit": "Nombre maximum de résultats"
        },
        "example": "Montre-moi les 10 meilleures prédictions avec une confiance > 65% pour demain"
    },
    
    "analyze_match": {
        "description": "Analyser un match spécifique en détail",
        "parameters": {
            "team1": "Première équipe",
            "team2": "Deuxième équipe",
            "match_id": "ID du match (optionnel)"
        },
        "example": "Analyse détaillée du match Bayern Munich vs Borussia Dortmund"
    },
    
    "find_value_bets": {
        "description": "Rechercher des paris à valeur ajoutée",
        "parameters": {
            "min_edge": "Edge minimum requis",
            "leagues": "Ligues à analyser",
            "date": "Période de recherche"
        },
        "example": "Trouve des value bets avec un edge > 8% pour ce weekend"
    }
}

RESPONSE_TEMPLATES = {
    "combo_success": """
✅ **Combiné créé avec succès !**

🎯 **Détails du Combiné:**
- **Type:** {combo_type}
- **Nombre de sélections:** {num_selections}
- **Cote totale:** {total_odds:.2f}
- **Mise recommandée:** {recommended_stake}
- **Valeur attendue:** {expected_value:+.1%}

📊 **Sélections:**
{selections_list}

💡 **Analyse:** Ce combiné présente un bon équilibre risque/rendement avec une probabilité de succès de {success_probability:.1%}.
    """,
    
    "predictions_success": """
🔮 **Prédictions trouvées: {count}**

📈 **Critères appliqués:**
- Confiance minimum: {min_confidence:.0%}
- Période: {period}
- Ligues: {leagues}

🎯 **Top Prédictions:**
{predictions_list}

💡 **Recommandation:** Concentrez-vous sur les prédictions avec confiance > 70% pour optimiser vos chances de succès.
    """,
    
    "portfolio_status": """
💰 **Statut du Portefeuille**

📊 **Métriques Clés:**
- **Bankroll Total:** €{total_bankroll:,.0f}
- **P&L Total:** €{total_pnl:+,.0f}
- **ROI:** {roi:+.1%}
- **Win Rate:** {win_rate:.1%}
- **Sharpe Ratio:** {sharpe_ratio:.2f}
- **Max Drawdown:** {max_drawdown:.1%}

📈 **Performance Récente:**
- Paris cette semaine: {recent_bets}
- Taux de succès: {recent_win_rate:.1%}

💡 **Recommandation:** {recommendation}
    """,
    
    "value_bets_success": """
💎 **Paris à Valeur Détectés: {count}**

🎯 **Critères:**
- Edge minimum: {min_edge:.1%}
- Période analysée: {period}

📊 **Meilleures Opportunités:**
{value_bets_list}

⚠️ **Important:** Vérifiez les cotes actuelles car elles peuvent évoluer rapidement.
    """,
    
    "error_response": """
❌ **Erreur de Traitement**

🔍 **Problème:** {error_message}

💡 **Suggestions:**
- Vérifiez l'orthographe des noms d'équipes
- Utilisez des formats de date reconnus (aujourd'hui, demain, ce weekend)
- Précisez les critères numériques (cotes, confiance, etc.)

🆘 **Besoin d'aide ?** Tapez "aide" pour voir les commandes disponibles.
    """
}

HELP_MESSAGES = {
    "general": """
🤖 **Assistant IA Bot Quantum Max**

**Commandes Principales:**
1. **Combinés** - Créer des paris combinés optimisés
2. **Prédictions** - Obtenir des pronostics haute confiance  
3. **Analyse** - Étudier des matchs en détail
4. **Portefeuille** - Suivre vos performances
5. **Value Bets** - Trouver des opportunités rentables

**Exemples:**
- "Crée un combiné avec 3 matchs d'aujourd'hui"
- "Top prédictions pour demain"
- "Analyse PSG vs Madrid"
- "Statut de mon portefeuille"

Tapez votre commande en langage naturel !
    """,
    
    "combo_help": """
📊 **Guide des Combinés**

**Types disponibles:**
- **Conservateur** - Faible risque, cotes modérées
- **Équilibré** - Bon compromis risque/rendement  
- **Agressif** - Haut rendement, risque élevé

**Paramètres:**
- Nombre de matchs (2-8)
- Cotes min/max par sélection
- Ligues spécifiques
- Période (aujourd'hui, demain, weekend)

**Exemples:**
- "Combiné conservateur avec 3 matchs PL"
- "Système agressif 5 sélections cotes > 1.8"
    """,
    
    "predictions_help": """
🔮 **Guide des Prédictions**

**Filtres disponibles:**
- **Confiance** - Niveau de certitude du modèle (50%-95%)
- **Ligue** - PL, FL1, BL1, SA, PD, CL, EL
- **Période** - Aujourd'hui, demain, semaine
- **Limite** - Nombre de résultats

**Types de prédictions:**
- Résultat (1X2)
- Over/Under buts
- Both Teams to Score
- Scores exacts

**Exemple:**
"Prédictions confiance > 75% pour Premier League demain"
    """
}

def get_example_commands(category: str = None) -> list:
    """Get example commands for a specific category or all."""
    if category and category in CHATBOT_EXAMPLES:
        return CHATBOT_EXAMPLES[category]
    
    all_examples = []
    for examples in CHATBOT_EXAMPLES.values():
        all_examples.extend(examples)
    
    return all_examples

def get_command_template(command: str) -> dict:
    """Get template for a specific command."""
    return COMMAND_TEMPLATES.get(command, {})

def format_response(template_name: str, **kwargs) -> str:
    """Format a response using a template."""
    template = RESPONSE_TEMPLATES.get(template_name, "")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"Erreur de formatage: paramètre manquant {e}"

def get_help_message(topic: str = "general") -> str:
    """Get help message for a specific topic."""
    return HELP_MESSAGES.get(topic, HELP_MESSAGES["general"])