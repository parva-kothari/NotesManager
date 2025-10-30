from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import pipeline
from datasets import Dataset
from rank_bm25 import BM25Okapi
import torch

app = Flask(__name__)
CORS(app)

print("Loading ML models...")

def init_db():
    conn = sqlite3.connect('notes.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            importance_score REAL DEFAULT 0.5,
            bm25_score REAL DEFAULT 0.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS ml_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            accuracy REAL,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized")

init_db()

# ============================================================================
# LOAD TRAINING DATASET FROM CSV
# ============================================================================

def load_training_data():
    csv_path = 'notes_training_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"Training dataset not found: {csv_path}")
        print("   Please ensure notes_training_dataset.csv is in the same directory")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded training dataset: {len(df)} examples")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

training_df = load_training_data()

# ============================================================================
# ML CLASSIFIER (SetFit + BART Zero-Shot)
# ============================================================================

class NotesClassifierAdvanced:
    def __init__(self, training_data=None):
        self.categories = ['academic', 'work', 'personal', 'health', 'finance', 'family']
        self.setfit_model = None
        self.zero_shot_model = None
        self.is_trained = False
        
        if training_data is not None and not training_data.empty:
            print("Initializing few-shot classifier (SetFit)...")
            self._init_setfit(training_data)
        else:
            print("No training data - using zero-shot only")
        
        print("Loading zero-shot classifier (BART-NLI)...")
        self._init_zero_shot()
    
    def _init_setfit(self, df):
        try:
            train_texts = df['content'].tolist()
            train_labels = df['category'].tolist()
            
            train_dataset = Dataset.from_dict({
                'text': train_texts,
                'label': train_labels
            })
            
            self.setfit_model = SetFitModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                labels=self.categories,
            )
            
            args = TrainingArguments(
                num_epochs=3,
                batch_size=16,
                learning_rate=2e-5,
                warmup_steps=10,
            )
            
            trainer = Trainer(
                model=self.setfit_model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
                metric="accuracy",
                column_mapping={"text": "text", "label": "label"}
            )
            
            print(f"Training SetFit on {len(train_texts)} examples...")
            trainer.train()
            
            accuracy = trainer.evaluate()
            print(f"SetFit trained! Accuracy: {accuracy.get('accuracy', 0):.2%}")
            
            self.is_trained = True
            
        except Exception as e:
            print(f"SetFit initialization failed: {e}")
            self.setfit_model = None
    
    def _init_zero_shot(self):
        try:
            self.zero_shot_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Zero-shot BART-NLI loaded")
        except Exception as e:
            print(f"Zero-shot model loading failed: {e}")
            self.zero_shot_model = None
    
    def classify_with_ranking(self, text):
        results = {
            'final_category': 'personal',
            'confidence': 0.5,
            'ranked_categories': [],
            'method_used': 'fallback'
        }
        
        if self.setfit_model and self.is_trained:
            try:
                predictions = self.setfit_model.predict([text])
                prediction = predictions[0] if isinstance(predictions, list) else predictions.item()
                
                probs = self.setfit_model.predict_proba([text])[0]
                
                ranked = sorted(
                    zip(self.categories, probs),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                results['final_category'] = prediction
                results['confidence'] = float(max(probs))
                results['ranked_categories'] = [(cat, float(prob)) for cat, prob in ranked]
                results['method_used'] = 'setfit'
                
                print(f"   SetFit: {prediction} ({max(probs):.2%})")
                return results
                
            except Exception as e:
                print(f"SetFit error: {e}")
        
        if self.zero_shot_model:
            try:
                output = self.zero_shot_model(
                    text,
                    candidate_labels=self.categories,
                    multi_label=False
                )
                
                ranked = list(zip(output['labels'], output['scores']))
                
                results['final_category'] = output['labels'][0]
                results['confidence'] = float(output['scores'][0])
                results['ranked_categories'] = [(cat, float(score)) for cat, score in ranked]
                results['method_used'] = 'zero_shot_nli'
                
                print(f"   Zero-Shot: {output['labels'][0]} ({output['scores'][0]:.2%})")
                return results
                
            except Exception as e:
                print(f"Zero-shot error: {e}")
        
        results['ranked_categories'] = [(cat, 1.0/len(self.categories)) for cat in self.categories]
        return results

classifier = NotesClassifierAdvanced(training_df)

# ============================================================================
# BM25 IMPORTANCE SCORING
# ============================================================================

class ImportanceScorer:
    def __init__(self, training_data=None):
        self.bm25 = None
        self.corpus = []
        self.importance_scores = []
        
        if training_data is not None and not training_data.empty:
            self.learn_from_training_data(training_data)
    
    def learn_from_training_data(self, df):
        self.corpus = [text.lower().split() for text in df['content'].tolist()]
        self.importance_scores = df['importance'].tolist()
        
        try:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"BM25 trained on {len(self.corpus)} examples")
        except Exception as e:
            print(f"BM25 error: {e}")
    
    def calculate_importance(self, note_content):
        if not self.bm25 or not self.importance_scores:
            return 0.5
        
        tokenized = note_content.lower().split()
        scores = self.bm25.get_scores(tokenized)
        
        if len(scores) == 0:
            return 0.5
        
        top_k = 5
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        similar_importances = [self.importance_scores[i] for i in top_indices if i < len(self.importance_scores)]
        
        if not similar_importances:
            return 0.5
        
        predicted_importance = np.mean(similar_importances)
        
        return float(np.clip(predicted_importance, 0.0, 1.0))
    
    def get_top_important_notes(self, query, note_ids, top_k=5):
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [note_ids[i] for i in top_indices if i < len(note_ids)]

importance_scorer = ImportanceScorer(training_df)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/notes', methods=['GET'])
def get_notes():
    category = request.args.get('category')
    sort_by = request.args.get('sort', 'date')
    
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = 'SELECT * FROM notes'
    params = []
    
    if category and category != 'all':
        query += ' WHERE category = ?'
        params.append(category)
    
    if sort_by == 'importance':
        query += ' ORDER BY importance_score DESC, created_at DESC'
    elif sort_by == 'access':
        query += ' ORDER BY access_count DESC, created_at DESC'
    else:
        query += ' ORDER BY created_at DESC'
    
    c.execute(query, params)
    notes = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify(notes)

@app.route('/api/notes', methods=['POST'])
def create_note():
    data = request.json
    content = data.get('content', '').strip()
    manual_category = data.get('category')
    
    if not content:
        return jsonify({'error': 'Content required'}), 400
    
    print(f"\nCreating note: '{content[:50]}...'")
    
    if manual_category:
        category = manual_category
        confidence = 1.0
        ranked_categories = [(manual_category, 1.0)]
        method = 'manual'
    else:
        classification = classifier.classify_with_ranking(content)
        category = classification['final_category']
        confidence = classification['confidence']
        ranked_categories = classification['ranked_categories']
        method = classification['method_used']
    
    importance = importance_scorer.calculate_importance(content)
    
    print(f"   Category: {category} ({confidence:.2%})")
    print(f"   Importance: {importance:.2f}")
    
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO notes (content, category, importance_score, bm25_score)
        VALUES (?, ?, ?, ?)
    ''', (content, category, importance, importance))
    
    note_id = c.lastrowid
    
    c.execute('''
        INSERT INTO ml_metrics (model_type, accuracy, confidence)
        VALUES (?, ?, ?)
    ''', (method, confidence, confidence))
    
    conn.commit()
    
    c.execute('SELECT * FROM notes WHERE id = ?', (note_id,))
    new_note = dict(c.fetchone())
    conn.close()
    
    return jsonify({
        'note': new_note,
        'classification': {
            'category': category,
            'confidence': confidence,
            'ranked_categories': ranked_categories[:3],
            'method': method
        },
        'importance_score': importance
    }), 201

@app.route('/api/notes/<int:note_id>', methods=['GET'])
def get_note(note_id):
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        UPDATE notes 
        SET access_count = access_count + 1,
            last_accessed = ?,
            importance_score = LEAST(importance_score + 0.05, 1.0)
        WHERE id = ?
    ''', (datetime.now().isoformat(), note_id))
    
    conn.commit()
    
    c.execute('SELECT * FROM notes WHERE id = ?', (note_id,))
    note = c.fetchone()
    
    if not note:
        conn.close()
        return jsonify({'error': 'Note not found'}), 404
    
    conn.close()
    return jsonify(dict(note))

@app.route('/api/notes/<int:note_id>', methods=['PUT'])
def update_note(note_id):
    data = request.json
    content = data.get('content')
    category = data.get('category')
    
    conn = sqlite3.connect('notes.db')
    c = conn.cursor()
    
    if content and category:
        c.execute('''
            UPDATE notes 
            SET content = ?, category = ?, updated_at = ?
            WHERE id = ?
        ''', (content, category, datetime.now().isoformat(), note_id))
    elif content:
        c.execute('''
            UPDATE notes 
            SET content = ?, updated_at = ?
            WHERE id = ?
        ''', (content, datetime.now().isoformat(), note_id))
    elif category:
        c.execute('''
            UPDATE notes 
            SET category = ?, updated_at = ?
            WHERE id = ?
        ''', (category, datetime.now().isoformat(), note_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Note updated'})

@app.route('/api/notes/<int:note_id>', methods=['DELETE'])
def delete_note(note_id):
    conn = sqlite3.connect('notes.db')
    c = conn.cursor()
    c.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Note deleted'})

@app.route('/api/search', methods=['GET'])
def search_notes():
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify([])
    
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT id FROM notes')
    note_ids = [row['id'] for row in c.fetchall()]
    
    top_ids = importance_scorer.get_top_important_notes(query, note_ids, top_k=10)
    
    if not top_ids:
        c.execute('''
            SELECT * FROM notes 
            WHERE content LIKE ? 
            ORDER BY importance_score DESC 
            LIMIT 10
        ''', (f'%{query}%',))
        results = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(results)
    
    placeholders = ','.join('?' * len(top_ids))
    c.execute(f'SELECT * FROM notes WHERE id IN ({placeholders})', top_ids)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify(results)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect('notes.db')
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM notes')
    total_notes = c.fetchone()[0]
    
    c.execute('SELECT category, COUNT(*) FROM notes GROUP BY category')
    category_counts = dict(c.fetchall())
    
    c.execute('SELECT AVG(importance_score) FROM notes')
    avg_importance = c.fetchone()[0] or 0
    
    c.execute('SELECT AVG(confidence) FROM ml_metrics WHERE timestamp > datetime("now", "-7 days")')
    avg_confidence = c.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        'total_notes': total_notes,
        'category_distribution': category_counts,
        'average_importance': round(avg_importance, 3),
        'ml_confidence_7d': round(avg_confidence, 3),
        'training_data_size': len(training_df) if training_df is not None else 0,
        'ml_stack': {
            'classification': 'SetFit (297 examples) + BART-NLI',
            'importance': 'BM25 regression',
            'research': 'NAACL 2024'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'Advanced Notes Manager API',
        'ml_models': {
            'setfit': 'active' if classifier.is_trained else 'inactive',
            'zero_shot': 'active' if classifier.zero_shot_model else 'inactive',
            'bm25': 'active' if importance_scorer.bm25 else 'inactive'
        },
        'training_data': len(training_df) if training_df is not None else 0
    })

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("NOTES MANAGER")
    print("="*60)
    print("Running on: http://0.0.0.0:5000")
    print("\nML Stack:")
    print("   ├─ SetFit Few-Shot")
    print("   ├─ BART Zero-Shot")
    print("   └─ BM25 Importance Ranking")
    print(f"\nTraining Data: {len(training_df) if training_df is not None else 0} examples")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
