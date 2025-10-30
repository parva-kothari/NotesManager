import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Modal,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useSafeAreaInsets } from 'react-native-safe-area-context'; 

const API_BASE_URL = 'http://192.168.193.158:5000/api';

const CATEGORIES = [
  { name: 'all', label: 'All Notes', icon: 'apps' },
  { name: 'academic', label: 'Academic', icon: 'school' },
  { name: 'work', label: 'Work', icon: 'briefcase' },
  { name: 'personal', label: 'Personal', icon: 'person' },
  { name: 'health', label: 'Health', icon: 'fitness' },
  { name: 'finance', label: 'Finance', icon: 'cash' },
  { name: 'family', label: 'Family', icon: 'people' },
];

export default function HomeScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets(); 
  
  const [notes, setNotes] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [newNoteContent, setNewNoteContent] = useState('');
  const [sidebarVisible, setSidebarVisible] = useState(false);

  useEffect(() => {
    loadNotes();
  }, [selectedCategory]);

  const loadNotes = async () => {
    try {
      let url = `${API_BASE_URL}/notes`;
      if (selectedCategory !== 'all') {
        url += `?category=${selectedCategory}`;
      }

      console.log('Loading notes from:', url);
      const response = await fetch(url);
      const data = await response.json();
      setNotes(data);
      console.log(`Loaded ${data.length} notes`);
    } catch (error) {
      console.error('Error loading notes:', error);
      Alert.alert('Connection Error', 'Cannot connect to backend. Check if server is running.');
    }
  };

  const addNote = async () => {
    if (!newNoteContent.trim()) {
      Alert.alert('Error', 'Please enter note content');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/notes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: newNoteContent }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Note added:', result);
        setNewNoteContent('');
        loadNotes();
        
        Alert.alert(
          'Note Created!',
          `Category: ${result.classification.category}\nConfidence: ${(result.classification.confidence * 100).toFixed(0)}%\nImportance: ${(result.importance_score * 100).toFixed(0)}%`,
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('Error adding note:', error);
      Alert.alert('Error', 'Failed to add note');
    }
  };

  const deleteNote = async (noteId) => {
    Alert.alert(
      'Delete Note',
      'Are you sure you want to delete this note?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await fetch(`${API_BASE_URL}/notes/${noteId}`, {
                method: 'DELETE',
              });
              loadNotes();
            } catch (error) {
              console.error('Error deleting note:', error);
            }
          },
        },
      ]
    );
  };

  const selectCategory = (category) => {
    setSelectedCategory(category);
    setSidebarVisible(false);
  };

  const getImportanceColor = (score) => {
    if (score >= 0.7) return '#ff4444';
    if (score >= 0.4) return '#ffaa00';
    return '#44ff44';
  };

  const getCategoryColor = (category) => {
    const colors = {
      academic: '#2196F3',
      work: '#9C27B0',
      personal: '#FF9800',
      health: '#4CAF50',
      finance: '#F44336',
      family: '#E91E63',
    };
    return colors[category] || '#666';
  };

  const getCategoryIcon = (category) => {
    const cat = CATEGORIES.find(c => c.name === category);
    return cat ? cat.icon : 'document';
  };

  return (
    <View style={styles.container}>  
      <StatusBar style="dark" />  
      
      <View style={[styles.header, { paddingTop: insets.top + 10 }]}>
        <TouchableOpacity onPress={() => setSidebarVisible(true)}>
          <Ionicons name="menu" size={28} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notes Manager</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity 
            onPress={() => router.push('/voice-checklist')}
            style={styles.headerButton}
          >
            <Ionicons name="mic" size={24} color="#FF5722" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Sidebar Modal */}
      <Modal
        visible={sidebarVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setSidebarVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setSidebarVisible(false)}
        >
          <View style={styles.sidebar}>
            <Text style={styles.sidebarTitle}>Categories</Text>
            
            <ScrollView showsVerticalScrollIndicator={false}>
              {CATEGORIES.map((category) => (
                <TouchableOpacity
                  key={category.name}
                  style={[
                    styles.categoryItem,
                    selectedCategory === category.name && styles.categoryItemActive,
                  ]}
                  onPress={() => selectCategory(category.name)}
                >
                  <View style={styles.categoryItemContent}>
                    <Ionicons
                      name={category.icon}
                      size={20}
                      color={selectedCategory === category.name ? '#fff' : '#666'}
                    />
                    <Text
                      style={[
                        styles.categoryText,
                        selectedCategory === category.name && styles.categoryTextActive,
                      ]}
                    >
                      {category.label}
                    </Text>
                  </View>
                  {selectedCategory === category.name && (
                    <Ionicons name="checkmark" size={20} color="#fff" />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>

            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setSidebarVisible(false)}
            >
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Main Content */}
      <ScrollView style={styles.content}>
        {/* Input Area */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Write your note here... (ML will auto-categorize)"
            placeholderTextColor="#999"
            value={newNoteContent}
            onChangeText={setNewNoteContent}
            multiline
            numberOfLines={3}
          />
          <TouchableOpacity style={styles.addButton} onPress={addNote}>
            <Ionicons name="add-circle" size={40} color="#2196F3" />
          </TouchableOpacity>
        </View>

        {/* Category Display */}
        <View style={styles.categoryTitleContainer}>
          <Ionicons
            name={getCategoryIcon(selectedCategory)}
            size={24}
            color={getCategoryColor(selectedCategory)}
          />
          <Text style={styles.categoryTitle}>
            {CATEGORIES.find(c => c.name === selectedCategory)?.label || 'All Notes'}
          </Text>
          <Text style={styles.notesCount}>({notes.length})</Text>
        </View>

        {/* Notes List */}
        <View style={styles.notesList}>
          {notes.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="document-text-outline" size={64} color="#ccc" />
              <Text style={styles.emptyText}>
                No notes yet. Add your first note!
              </Text>
              <Text style={styles.emptySubtext}>
                The ML model will automatically categorize it
              </Text>
            </View>
          ) : (
            notes.map((note) => (
              <View key={note.id} style={styles.noteCard}>
                <View style={styles.noteHeader}>
                  <View style={styles.noteTags}>
                    <View
                      style={[
                        styles.categoryBadge,
                        { backgroundColor: getCategoryColor(note.category) },
                      ]}
                    >
                      <Text style={styles.categoryBadgeText}>
                        {note.category}
                      </Text>
                    </View>
                    <View
                      style={[
                        styles.importanceBadge,
                        { backgroundColor: getImportanceColor(note.importance_score) },
                      ]}
                    >
                      <Text style={styles.importanceBadgeText}>
                        {(note.importance_score * 100).toFixed(0)}%
                      </Text>
                    </View>
                  </View>
                  <TouchableOpacity onPress={() => deleteNote(note.id)}>
                    <Ionicons name="trash-outline" size={20} color="#ff4444" />
                  </TouchableOpacity>
                </View>

                <Text style={styles.noteContent}>{note.content}</Text>

                <View style={styles.noteFooter}>
                  <Text style={styles.noteTimestamp}>
                    {new Date(note.created_at).toLocaleString()}
                  </Text>
                  {note.access_count > 0 && (
                    <View style={styles.accessBadge}>
                      <Ionicons name="eye" size={12} color="#666" />
                      <Text style={styles.accessCount}>{note.access_count}</Text>
                    </View>
                  )}
                </View>
              </View>
            ))
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
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
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  headerButton: {
    padding: 4,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  input: {
    flex: 1,
    fontSize: 16,
    minHeight: 60,
    maxHeight: 120,
    textAlignVertical: 'top',
  },
  addButton: {
    marginLeft: 8,
  },
  categoryTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 8,
  },
  categoryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  notesCount: {
    fontSize: 14,
    color: '#999',
  },
  notesList: {
    flex: 1,
  },
  emptyState: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#ccc',
    textAlign: 'center',
    marginTop: 8,
  },
  noteCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  noteHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  noteTags: {
    flexDirection: 'row',
    gap: 8,
  },
  categoryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  categoryBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  importanceBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  importanceBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  noteContent: {
    fontSize: 16,
    color: '#333',
    marginBottom: 8,
    lineHeight: 22,
  },
  noteFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  noteTimestamp: {
    fontSize: 12,
    color: '#999',
  },
  accessBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  accessCount: {
    fontSize: 12,
    color: '#666',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  sidebar: {
    width: '75%',
    height: '100%',
    backgroundColor: '#fff',
    padding: 20,
  },
  sidebarTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  categoryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 8,
    marginBottom: 8,
    backgroundColor: '#f5f5f5',
  },
  categoryItemActive: {
    backgroundColor: '#2196F3',
  },
  categoryItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  categoryText: {
    fontSize: 16,
    color: '#333',
  },
  categoryTextActive: {
    color: '#fff',
    fontWeight: 'bold',
  },
  closeButton: {
    marginTop: 20,
    padding: 16,
    backgroundColor: '#ff4444',
    borderRadius: 8,
    alignItems: 'center',
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
