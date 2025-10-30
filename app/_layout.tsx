import { Stack } from 'expo-router';
import { useEffect } from 'react';

export default function RootLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: '#f5f5f5' },
      }}
    >
      <Stack.Screen name="index" options={{ title: 'Notes Manager' }} />
      <Stack.Screen 
        name="checklist" 
        options={{ 
          title: 'Checklist',
          presentation: 'modal'
        }} 
      />
    </Stack>
  );
}
