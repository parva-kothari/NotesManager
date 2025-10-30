import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  Modal,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  useSpeechRecognitionEvent,
  ExpoSpeechRecognitionModule,
} from 'expo-speech-recognition';

const API_BASE_URL = 'http://192.168.193.158:5001/api/voice';

export default function VoiceChecklistScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState({ checklist: {}, ml_performance: {} });
  const [isListening, setIsListening] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  const [wakeWordDetected, setWakeWordDetected] = useState(false);
  const [filter, setFilter] = useState('active');
  const [showAddModal, setShowAddModal] = useState(false);
  const [newItemText, setNewItemText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Ensemble ML display
  const [ensembleScores, setEnsembleScores] = useState<any>(null);
  const [methodBreakdown, setMethodBreakdown] = useState<any>(null);
  const [showEnsembleModal, setShowEnsembleModal] = useState(false);
  
  const commandTimeout = useRef<NodeJS.Timeout | null>(null);
  const lastCommand = useRef('');
  const shouldAutoRestart = useRef(true);
  const processingCommand = useRef(false);

  useEffect(() => {
    loadItems();
    loadStats();
  }, [filter]);

  useSpeechRecognitionEvent('start', () => {
    console.log('Speech started');
    setIsListening(true);
  });

  useSpeechRecognitionEvent('end', () => {
    console.log('Speech ended');
    setIsListening(false);
    
    if (commandTimeout.current) {
      clearTimeout(commandTimeout.current);
      commandTimeout.current = null;
    }
    
    if (shouldAutoRestart.current && !processingCommand.current) {
      console.log('Auto-restarting...');
      setTimeout(() => startListening(), 1000);
    } else {
      setWakeWordDetected(false);
    }
  });

  useSpeechRecognitionEvent('result', (event) => {
    if (!event.results || event.results.length === 0) return;
    
    const result = event.results[0];
    const text = result.transcript || '';
    
    setRecognizedText(text);
    
    if (text.toLowerCase().includes('command')) {
      if (!wakeWordDetected) {
        console.log('Wake word detected!');
        setWakeWordDetected(true);
        setRecognizedText('Wake word detected! Continue speaking...');
      }
      return;
    }
    
    if (wakeWordDetected && !processingCommand.current) {
      if (commandTimeout.current) {
        clearTimeout(commandTimeout.current);
      }
      
      commandTimeout.current = setTimeout(() => {
        const finalCommand = text.trim();
        
        if (finalCommand && finalCommand !== lastCommand.current) {
          lastCommand.current = finalCommand;
          processingCommand.current = true;
          processVoiceCommand(finalCommand);
        }
      }, 1500);
    }
  });

  useSpeechRecognitionEvent('error', (event) => {
    console.error('Speech error:', event.error);
    setIsListening(false);
    setWakeWordDetected(false);
    processingCommand.current = false;
    
    if (commandTimeout.current) {
      clearTimeout(commandTimeout.current);
      commandTimeout.current = null;
    }
  });

  const startListening = async () => {
    try {
      shouldAutoRestart.current = true;
      processingCommand.current = false;
      setWakeWordDetected(false);
      setRecognizedText('');
      lastCommand.current = '';
      
      if (commandTimeout.current) {
        clearTimeout(commandTimeout.current);
        commandTimeout.current = null;
      }
      
      await ExpoSpeechRecognitionModule.start({
        lang: 'en-US',
        interimResults: true,
        continuous: true,
        maxAlternatives: 1,
      });
      
      console.log('Started listening');
    } catch (error) {
      console.error('Failed to start:', error);
      Alert.alert('Error', 'Failed to start voice recognition');
    }
  };

  const stopListening = async () => {
    try {
      shouldAutoRestart.current = false;
      processingCommand.current = false;
      
      if (commandTimeout.current) {
        clearTimeout(commandTimeout.current);
        commandTimeout.current = null;
      }
      
      await ExpoSpeechRecognitionModule.stop();
      
      setWakeWordDetected(false);
      setRecognizedText('');
      lastCommand.current = '';
      
      console.log('Stopped listening');
    } catch (error) {
      console.error('Failed to stop:', error);
    }
  };

  const processVoiceCommand = async (command: string) => {
    setIsProcessing(true);
    setRecognizedText('Processing with Ensemble ML (3 methods)...');
    
    try {
      const response = await fetch(`${API_BASE_URL}/process-command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command_text: command }),
      });

      const result = await response.json();
      
      console.log('Ensemble result:', result);
      
      // Save ensemble data for display
      setEnsembleScores(result.ensemble_scores);
      setMethodBreakdown(result.method_breakdown);
      
      if (result.success) {
        const icon = result.action === 'add' ? '' : '';
        setRecognizedText(
          `${icon} ${result.message}\n` +
          `Confidence: ${(result.confidence * 100).toFixed(0)}%\n` +
          `Method: Ensemble (3 models)`
        );
        
        await loadItems();
        await loadStats();
        
        // Auto-clear after 3 seconds
        setTimeout(() => {
          setRecognizedText('');
          setWakeWordDetected(false);
          processingCommand.current = false;
          lastCommand.current = '';
        }, 3000);
        
      } else {
        setRecognizedText(`${result.message}`);
        setTimeout(() => {
          setRecognizedText('');
          setWakeWordDetected(false);
          processingCommand.current = false;
        }, 2000);
      }
      
    } catch (error) {
      console.error('Voice processing error:', error);
      setRecognizedText('Processing failed');
      setTimeout(() => {
        setRecognizedText('');
        setWakeWordDetected(false);
        processingCommand.current = false;
      }, 2000);
    }

    setIsProcessing(false);
  };

  const loadItems = async () => {
    try {
      let url = `${API_BASE_URL}/checklist/items`;
      if (filter === 'active') url += '?completed=false';
      else if (filter === 'completed') url += '?completed=true';
      
      const response = await fetch(url);
      const data = await response.json();
      setItems(data);
    } catch (error) {
      console.error('Error loading items:', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const addItemManually = async () => {
    if (!newItemText.trim()) {
      Alert.alert('Error', 'Please enter item text');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/checklist/items`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_text: newItemText }),
      });

      if (response.ok) {
        setNewItemText('');
        setShowAddModal(false);
        loadItems();
        loadStats();
      }
    } catch (error) {
      console.error('Add item error:', error);
      Alert.alert('Error', 'Failed to add item');
    }
  };

  const toggleItemCompletion = async (item: any) => {
    try {
      await fetch(`${API_BASE_URL}/checklist/items/${item.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ completed: !item.completed }),
      });
      loadItems();
      loadStats();
    } catch (error) {
      console.error('Toggle error:', error);
    }
  };

  const deleteItem = async (item: any) => {
    Alert.alert('Delete Item', `Delete "${item.item_text}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await fetch(`${API_BASE_URL}/checklist/items/${item.id}`, {
              method: 'DELETE',
            });
            loadItems();
            loadStats();
          } catch (error) {
            console.error('Delete error:', error);
          }
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      
      <View style={[styles.header, { paddingTop: insets.top + 10 }]}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={28} color="#333" />
        </TouchableOpacity>
        <Text style={styles.title}>Voice Checklist</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity 
            onPress={() => setShowEnsembleModal(true)}
            style={styles.mlBadge}
          >
            <Ionicons name="analytics" size={16} color="#4CAF50" />
            <Text style={styles.mlBadgeText}>ML</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => setShowAddModal(true)}>
            <Ionicons name="add-circle" size={28} color="#4CAF50" />
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.stats}>
        <View style={styles.statBox}>
          <Text style={styles.statNum}>{stats.checklist?.pending || 0}</Text>
          <Text style={styles.statLabel}>Pending</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={[styles.statNum, {color: '#4CAF50'}]}>
            {stats.checklist?.completed || 0}
          </Text>
          <Text style={styles.statLabel}>Done</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statNum}>{stats.ml_performance?.total_commands || 0}</Text>
          <Text style={styles.statLabel}>Commands</Text>
        </View>
      </View>

      <ScrollView style={styles.content}>
        <View style={[styles.voiceBox, isListening && styles.voiceBoxActive]}>
          <View style={styles.voiceHeader}>
            <Ionicons 
              name={isListening ? 'mic' : 'mic-off'} 
              size={24} 
              color={isListening ? '#ff4444' : '#999'} 
            />
            <Text style={styles.voiceTitle}>
              {isListening ? 'Listening' : 'Voice Control (Stopped)'}
            </Text>
          </View>

          <View style={styles.helpBox}>
            <Text style={styles.helpTitle}>Ensemble ML (3 Methods):</Text>
            <Text style={styles.helpText}>• Rule-based: Keyword matching</Text>
            <Text style={styles.helpText}>• Semantic: Sentence-BERT</Text>
            <Text style={styles.helpText}>• Voting: Confidence agreement</Text>
          </View>

          {recognizedText && (
            <View style={styles.transcript}>
              <Text style={styles.transcriptText}>{recognizedText}</Text>
            </View>
          )}

          <TouchableOpacity
            style={[styles.micButton, isListening && styles.micButtonActive]}
            onPress={isListening ? stopListening : startListening}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Ionicons 
                name={isListening ? 'stop-circle' : 'mic-circle'} 
                size={48} 
                color="#fff" 
              />
            )}
            <Text style={styles.micButtonText}>
              {isProcessing ? 'Processing...' : isListening ? 'Stop' : 'Start'}
            </Text>
          </TouchableOpacity>

          {isListening && (
            <View style={styles.statusBox}>
              <Text style={styles.statusText}>
                {wakeWordDetected ? 'Waiting for command...' : 'Say "Command"...'}
              </Text>
            </View>
          )}
        </View>

        <View style={styles.filters}>
          {['all', 'active', 'completed'].map(f => (
            <TouchableOpacity
              key={f}
              style={[styles.filterBtn, filter === f && styles.filterBtnActive]}
              onPress={() => setFilter(f)}
            >
              <Text style={[styles.filterText, filter === f && styles.filterTextActive]}>
                {f}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.list}>
          {items.length === 0 ? (
            <View style={styles.empty}>
              <Ionicons name="list-outline" size={64} color="#ccc" />
              <Text style={styles.emptyText}>No items</Text>
              <Text style={styles.emptySubtext}>
                Say "Command add [item]" or tap +
              </Text>
            </View>
          ) : (
            items.map((item: any) => (
              <View key={item.id} style={styles.item}>
                <TouchableOpacity onPress={() => toggleItemCompletion(item)}>
                  <Ionicons
                    name={item.completed ? 'checkbox' : 'square-outline'}
                    size={24}
                    color={item.completed ? '#4CAF50' : '#999'}
                  />
                </TouchableOpacity>
                <Text style={[styles.itemText, item.completed && styles.itemDone]}>
                  {item.item_text}
                </Text>
                <TouchableOpacity onPress={() => deleteItem(item)}>
                  <Ionicons name="trash-outline" size={20} color="#ff4444" />
                </TouchableOpacity>
              </View>
            ))
          )}
        </View>
      </ScrollView>

      {/* Ensemble ML Modal */}
      <Modal visible={showEnsembleModal} transparent animationType="slide">
        <View style={styles.modalBack}>
          <View style={styles.ensembleModal}>
            <Text style={styles.modalTitle}>Ensemble ML Breakdown</Text>
            
            {ensembleScores && (
              <>
                <Text style={styles.sectionTitle}>Final Scores:</Text>
                <View style={styles.scoreBox}>
                  <Text style={styles.scoreText}>
                    Add: {(ensembleScores.add * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.scoreText}>
                    Remove: {(ensembleScores.remove * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.scoreText}>
                    Unknown: {(ensembleScores.unknown * 100).toFixed(1)}%
                  </Text>
                </View>
              </>
            )}
            
            <TouchableOpacity
              style={styles.closeBtn}
              onPress={() => setShowEnsembleModal(false)}
            >
              <Text style={styles.closeBtnText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Add Item Modal */}
      <Modal visible={showAddModal} transparent animationType="slide">
        <View style={styles.modalBack}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>Add Item</Text>
            <TextInput
              style={styles.modalInput}
              placeholder="e.g., Buy Milk"
              value={newItemText}
              onChangeText={setNewItemText}
              autoFocus
            />
            <View style={styles.modalBtns}>
              <TouchableOpacity
                style={[styles.modalBtn, styles.modalBtnCancel]}
                onPress={() => {
                  setShowAddModal(false);
                  setNewItemText('');
                }}
              >
                <Text style={styles.modalBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalBtn, styles.modalBtnAdd]}
                onPress={addItemManually}
              >
                <Text style={[styles.modalBtnText, {color: '#fff'}]}>Add</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  title: { fontSize: 20, fontWeight: 'bold', color: '#333' },
  headerRight: { flexDirection: 'row', gap: 12, alignItems: 'center' },
  mlBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  mlBadgeText: { fontSize: 11, fontWeight: '600', color: '#4CAF50' },
  stats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
    backgroundColor: '#E3F2FD',
  },
  statBox: { alignItems: 'center' },
  statNum: { fontSize: 24, fontWeight: 'bold', color: '#2196F3' },
  statLabel: { fontSize: 12, color: '#666', marginTop: 4 },
  content: { flex: 1 },
  voiceBox: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 20,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#ddd',
  },
  voiceBoxActive: {
    borderColor: '#ff4444',
    backgroundColor: '#fff5f5',
  },
  voiceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  voiceTitle: { fontSize: 16, fontWeight: 'bold' },
  helpBox: {
    backgroundColor: '#f5f5f5',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  helpTitle: { fontSize: 13, fontWeight: 'bold', marginBottom: 8 },
  helpText: { fontSize: 12, color: '#666', marginBottom: 4 },
  transcript: {
    backgroundColor: '#E8F5E9',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  transcriptText: { fontSize: 14, fontWeight: '600', color: '#333' },
  micButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2196F3',
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  micButtonActive: { backgroundColor: '#ff4444' },
  micButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  statusBox: {
    backgroundColor: '#FFF9C4',
    padding: 10,
    borderRadius: 8,
    marginTop: 12,
    alignItems: 'center',
  },
  statusText: { fontSize: 12, color: '#F57F17', fontWeight: '600' },
  filters: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 0,
    padding: 8,
    borderRadius: 8,
  },
  filterBtn: {
    flex: 1,
    padding: 10,
    alignItems: 'center',
    borderRadius: 6,
  },
  filterBtnActive: { backgroundColor: '#2196F3' },
  filterText: { color: '#666', fontWeight: '600', textTransform: 'capitalize' },
  filterTextActive: { color: '#fff' },
  list: { padding: 16, paddingTop: 0 },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  itemText: { flex: 1, fontSize: 16, color: '#333' },
  itemDone: { textDecorationLine: 'line-through', color: '#999' },
  empty: { padding: 40, alignItems: 'center' },
  emptyText: { fontSize: 18, color: '#999', marginTop: 16, fontWeight: '600' },
  emptySubtext: { fontSize: 14, color: '#ccc', marginTop: 8, textAlign: 'center' },
  modalBack: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modal: {
    backgroundColor: '#fff',
    width: '85%',
    padding: 24,
    borderRadius: 16,
  },
  ensembleModal: {
    backgroundColor: '#fff',
    width: '90%',
    padding: 24,
    borderRadius: 16,
    maxHeight: '80%',
  },
  modalTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 20 },
  modalInput: {
    backgroundColor: '#f5f5f5',
    padding: 16,
    borderRadius: 12,
    fontSize: 16,
    marginBottom: 20,
  },
  modalBtns: { flexDirection: 'row', gap: 12 },
  modalBtn: { flex: 1, padding: 16, borderRadius: 12, alignItems: 'center' },
  modalBtnCancel: { backgroundColor: '#f5f5f5' },
  modalBtnAdd: { backgroundColor: '#4CAF50' },
  modalBtnText: { fontSize: 16, fontWeight: '600', color: '#666' },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  scoreBox: {
    backgroundColor: '#E8F5E9',
    padding: 12,
    borderRadius: 8,
  },
  scoreText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  methodBox: {
    backgroundColor: '#E3F2FD',
    padding: 12,
    borderRadius: 8,
  },
  methodText: {
    fontSize: 13,
    color: '#333',
    marginBottom: 4,
  },
  closeBtn: {
    backgroundColor: '#2196F3',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
  },
  closeBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
