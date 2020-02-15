import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import AppContainer from './AppContainer';


export default function App() {
  init();
  return (
    <View style={styles.container}>
      <AppContainer/>
    </View>
  );
}

function init() {
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
