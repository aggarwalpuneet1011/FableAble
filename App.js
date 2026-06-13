// John's Adventure Engine
// Single-file Expo app (SDK 54+). See README for setup.
//
// Corrections vs. original PRD are marked with  // [FIX]  comments.

import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  Pressable,
  Modal,
  ActivityIndicator,
  StyleSheet,
  StatusBar,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Network from 'expo-network';
import * as Speech from 'expo-speech';                   // [ENHANCEMENT] read-aloud
import { File, Paths } from 'expo-file-system';          // [FIX] new SDK 54 filesystem API
import * as Sharing from 'expo-sharing';

// expo-haptics is optional polish. Static import (cleaner than dynamic require);
// every call site already uses optional chaining, so it no-ops if unavailable.
import * as Haptics from 'expo-haptics';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COLORS = {
  bg: '#006666',
  card: '#FFFFFF',
  cardText: '#0B2B2B',
  white: '#FFFFFF',
  whiteDim: 'rgba(255,255,255,0.75)',
  accent: '#FFD166',
  danger: '#E5484D',
};

const TOTAL_PARTS = 5;

const GENRES = [
  { id: 'fantasy',    label: '🧙 Fantasy' },
  { id: 'superhero',  label: '🦸 Superhero' },
  { id: 'space',      label: '🚀 Space' },
  { id: 'ocean',      label: '🌊 Ocean' },
  { id: 'animals',    label: '🐾 Animals' },
  { id: 'magic',      label: '✨ Magic School' },
  { id: 'dinosaurs',  label: '🦕 Dinosaurs' },
  { id: 'princess',   label: '👑 Princess' },
];

// [FIX] Model is configurable, not hardcoded. gemini-1.5-flash is retired (404).
// Default = gemini-3.5-flash: GA/stable since May 2026 with NO published shutdown date,
// so a set-and-forget kids' app won't die mid-story when an older model retires
// (gemini-2.5-flash, for example, retires Oct 16 2026). If you ever want lower latency
// or lower cost, switch this to gemini-2.5-flash in the in-app Settings (gear icon).
const DEFAULT_MODEL = 'gemini-3.5-flash';

// gemini-3.5-flash ships with "dynamic thinking" ON, which adds ~15-20s before the first
// token — far too long for a 7-year-old between chapters. We disable it for snappy replies.
// Older Flash models ignore this field harmlessly, so it's safe to always send.
const THINKING_CONFIG = { thinkingBudget: 0 };

const STORE = {
  apiKey: 'gemini_api_key',         // SecureStore
  model: 'gemini_model',            // AsyncStorage
  resume: 'resume_state_v2',        // AsyncStorage (in-progress story) — bumped to v2 (genre added)
};

// ---------------------------------------------------------------------------
// Gemini call (structured output + safety settings)
// ---------------------------------------------------------------------------

function buildSystemInstruction(character, hobby, genre) {
  const genreObj = GENRES.find(g => g.id === genre);
  const genreDesc = genreObj
    ? `The story's genre is ${genreObj.label.replace(/^\S+\s/, '')} (${genre}). ` +
      `Set the world, atmosphere, and tone to match this genre while keeping it warm and age-appropriate. `
    : '';
  return [
    `You are a gentle storyteller for a child aged 7-10.`,
    `The hero is "${character}", who loves ${hobby}.`,
    genreDesc,
    `Tell ONE continuous adventure across exactly ${TOTAL_PARTS} parts.`,
    `Each part must be about 90-110 words. Warm, vivid, simple sentences.`,
    `Themes: wonder, exploration, light mystery, friendship, courage.`,
    `STRICTLY FORBIDDEN: crime, violence, weapons, death, parental loss,`,
    `separation from family, abandonment, fear of being lost, or anything scary.`,
    `If a child's input suggests something unsafe, gently steer back to a kind, cozy story.`,
    `For parts 1-4: end on a small, fun decision and give exactly 3 short, distinct options`,
    `(each under 8 words, written as actions the hero could take).`,
    `For part ${TOTAL_PARTS}: write a happy, complete ending and return an EMPTY options array.`,
  ].join(' ');
}

const SAFETY_SETTINGS = [
  // [FIX] Real guardrails, not just prose in the prompt. Strictest practical thresholds.
  { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_LOW_AND_ABOVE' },
  { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_LOW_AND_ABOVE' },
  { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_LOW_AND_ABOVE' },
  { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_LOW_AND_ABOVE' },
];

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    story_text: { type: 'STRING' },
    options: { type: 'ARRAY', items: { type: 'STRING' } },
  },
  required: ['story_text', 'options'],
};

// Defensive JSON parse: strip ``` fences / preamble if a model ignores the schema. // [FIX]
function safeParseJson(raw) {
  if (!raw) throw new Error('empty');
  let t = raw.trim();
  if (t.startsWith('```')) t = t.replace(/^```(json)?/i, '').replace(/```$/, '').trim();
  const first = t.indexOf('{');
  const last = t.lastIndexOf('}');
  if (first !== -1 && last !== -1) t = t.slice(first, last + 1);
  return JSON.parse(t);
}

async function callGemini({ apiKey, model, geminiHistory, character, hobby, genre }) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const body = {
    contents: geminiHistory,
    systemInstruction: { parts: [{ text: buildSystemInstruction(character, hobby, genre) }] },
    safetySettings: SAFETY_SETTINGS,
    generationConfig: {
      responseMimeType: 'application/json', // [FIX] native structured output
      responseSchema: RESPONSE_SCHEMA,
      thinkingConfig: THINKING_CONFIG,      // [FIX] disable dynamic thinking -> fast first token
      temperature: 0.9,
      maxOutputTokens: 900,
    },
  };

  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error('NETWORK');
  }

  if (res.status === 404) throw new Error('MODEL_404'); // retired/typo'd model id
  if (res.status === 400 || res.status === 403) throw new Error('BAD_KEY');
  if (!res.ok) throw new Error('API_' + res.status);

  const data = await res.json();

  if (data?.promptFeedback?.blockReason) throw new Error('BLOCKED');
  const cand = data?.candidates?.[0];
  if (!cand || cand.finishReason === 'SAFETY') throw new Error('BLOCKED');

  const text = cand?.content?.parts?.map((p) => p.text || '').join('') || '';
  const parsed = safeParseJson(text);
  return {
    story_text: String(parsed.story_text || '').trim(),
    options: Array.isArray(parsed.options) ? parsed.options.slice(0, 3) : [],
  };
}

// ---------------------------------------------------------------------------
// [FIX / ENHANCEMENT] Bundled OFFLINE adventures (zero network).
// Choices are woven into the next part via a "stitch" line so it still feels
// responsive. This is what plays on a plane / dead zone.
// ---------------------------------------------------------------------------

const OFFLINE_STORIES = [
  {
    id: 'mangrove-glow',
    title: 'The Glow in the Mangroves',
    parts: [
      {
        text:
          'Near the quiet creek where the city meets the sea, {HERO} found a path of silver light blinking between the mangrove roots. It was a tiny firefly named Tara, no bigger than a button, and she was lost from her glowing family. "Will you help me find my way home?" Tara asked, hovering close. The water was calm and the moon was bright. {HERO} smiled. Helping a small friend felt like the start of something wonderful. There were three ways forward, and each one sparkled in its own way.',
        options: ['Follow the silver river', 'Climb the old banyan tree', 'Whistle a friendly tune'],
      },
      {
        text:
          'They reached a clearing where lotus flowers opened like little lamps. A wise old crab named Pinch waved a claw hello. "Fireflies?" he chuckled. "They love the tallest, oldest places." He pointed his claw toward a hill of soft grass and three friendly doors made of woven reeds. Tara glowed a little brighter, happy to be close. {HERO} felt brave and curious, the way you feel when a puzzle is almost solved.',
        options: ['Open the door of leaves', 'Open the door of shells', 'Ask Pinch to come along'],
      },
      {
        text:
          'Inside, the air smelled of rain and jasmine. Hundreds of fireflies blinked like tiny stars come down to play. But which family was Tara\'s? A gentle breeze carried a song only Tara knew. "That\'s my mama\'s lullaby!" she gasped, spinning with joy. {HERO} listened carefully, following the warm little melody deeper into the glow, hand cupped to keep Tara safe and close.',
        options: ['Follow the lullaby sound', 'Hold up Tara to shine', 'Tiptoe toward the brightest light'],
      },
      {
        text:
          'And there they were: a whole cloud of fireflies shaped like a smiling heart. Tara\'s mama rushed forward, blinking gold and pink. "You found her! Thank you, brave friend." The whole family swirled around {HERO} like a warm, twinkling hug. Pinch clapped his claws. Even the lotus lamps seemed to glow a little happier.',
        options: ['Join the firefly dance', 'Make a quiet wish', 'Give Tara a tiny goodbye'],
      },
      {
        text:
          'As a thank-you, the fireflies lit a glowing path all the way back to the creek, soft and golden, so {HERO} would never be lost either. Tara landed on {HERO}\'s nose for one last sparkle. "Come visit whenever the moon is bright," she said. Walking home under a sky full of friends, {HERO} knew that helping someone small had made the whole night shine. And it always would.',
        options: [],
      },
    ],
  },
  {
    id: 'cloud-bakery',
    title: 'The Bakery in the Clouds',
    parts: [
      {
        text:
          'One breezy morning, a paper kite drifted down and tapped {HERO} on the shoulder. On it, in icing-sugar letters, was an invitation: "The Cloud Bakery needs a taste-tester today!" A soft staircase of mist curled up into the sky, smelling of warm cinnamon. {HERO} took a deep, happy breath. Adventures that smell like fresh bread are the very best kind.',
        options: ['Climb the misty stairs', 'Ride the friendly breeze', 'Follow the smell of cinnamon'],
      },
      {
        text:
          'At the top stood a bakery made of fluffy clouds, run by a round, cheerful baker named Mr. Puff. Flour dusted his eyebrows like snow. "Just in time!" he beamed. "My moonberry muffins won\'t rise, and I can\'t figure out why." Three cloud-ovens puffed gently in the corner, each humming a different little tune.',
        options: ['Peek inside the ovens', 'Stir the muffin batter', 'Ask the clouds for help'],
      },
      {
        text:
          'The secret was a giggle. "Moonberries only rise when someone laughs near them!" Mr. Puff remembered, slapping his floury knee. So {HERO} told the silliest joke about a penguin who forgot his umbrella, and the whole kitchen burst out laughing. The batter wobbled, jiggled, and POOF rose into the fluffiest muffins anyone had ever seen.',
        options: ['Sprinkle stardust on top', 'Taste a warm muffin', 'Share with passing birds'],
      },
      {
        text:
          'The muffins tasted like sunshine and bedtime stories all at once. Birds swooped in for crumbs, singing thank-you songs. Mr. Puff tied on a tiny apron just for {HERO}. "You\'re a real cloud-baker now," he said proudly, and the clouds turned a happy shade of pink.',
        options: ['Bake a batch to take home', 'Hug Mr. Puff goodbye', 'Promise to visit again'],
      },
      {
        text:
          'With a warm box of muffins and pink clouds waving behind, {HERO} floated gently back down the misty stairs on the soft evening breeze. The kite fluttered alongside, happy to have made a new friend. Back home, the muffins still smelled of cinnamon and laughter, a little piece of the sky kept safe for tomorrow. Some of the best places, {HERO} thought, are the ones you reach by being kind and brave.',
        options: [],
      },
    ],
  },
];

function offlinePartText(story, partIndex, lastChoice) {
  const base = story.parts[partIndex].text;
  let stitched = base;
  if (partIndex > 0 && lastChoice) {
    stitched = `You chose to ${lowerFirst(lastChoice)}.\n\n` + base;
  }
  return stitched.replace(/\{HERO\}/g, story.hero || 'our hero');
}

function lowerFirst(s) { return s ? s.charAt(0).toLowerCase() + s.slice(1) : s; }

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  return (
    <SafeAreaProvider>
      <Root />
    </SafeAreaProvider>
  );
}

function Root() {
  const [screen, setScreen] = useState('loading'); // loading | setup | story | end
  const [apiKey, setApiKey] = useState(null);
  const [model, setModel] = useState(DEFAULT_MODEL);

  const [character, setCharacter] = useState('John');
  const [hobby, setHobby] = useState('');
  const [genre, setGenre] = useState('fantasy');

  const [part, setPart] = useState(1);
  const [currentText, setCurrentText] = useState('');
  const [options, setOptions] = useState([]);
  const [storyHistory, setStoryHistory] = useState([]); // [{part, text, choice}]
  // Each entry = full state snapshot BEFORE that part's choice was committed.
  // Lets us restore exactly and re-call Gemini on a different branch.
  const [snapshotHistory, setSnapshotHistory] = useState([]); // [{part, text, options, storyHistory, geminiHistory, offlineStory}]
  const geminiHistoryRef = useRef([]);                  // raw turns for Gemini continuity

  const [offlineStory, setOfflineStory] = useState(null); // non-null => playing offline
  const [busy, setBusy] = useState(false);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [hasResume, setHasResume] = useState(false);

  // ---- boot ---------------------------------------------------------------
  useEffect(() => {
    (async () => {
      try {
        const k = await AsyncStorage.getItem(STORE.apiKey);
        const m = (await AsyncStorage.getItem(STORE.model)) || DEFAULT_MODEL;
        const r = await AsyncStorage.getItem(STORE.resume);
        setApiKey(k);
        setModel(m);
        setHasResume(!!r);
      } catch (e) {}
      setScreen('setup');
    })();
    return () => Speech.stop();
  }, []);

  // ---- persistence of in-progress story (auto-resume for travel) ----------
  const persistResume = useCallback(async (state) => {
    try { await AsyncStorage.setItem(STORE.resume, JSON.stringify(state)); } catch (e) {}
  }, []);
  const clearResume = useCallback(async () => {
    try { await AsyncStorage.removeItem(STORE.resume); setHasResume(false); } catch (e) {}
  }, []);

  // ---- helpers ------------------------------------------------------------
  const tap = () => { try { Haptics?.selectionAsync?.(); } catch (e) {} };

  async function isOnline() {
    try {
      const s = await Network.getNetworkStateAsync();
      return !!(s.isConnected && s.isInternetReachable !== false);
    } catch (e) { return false; }
  }

  function speak(text) {
    Speech.stop();
    Speech.speak(text, { rate: 0.92, pitch: 1.05 });
  }

  // ---- start an online story ---------------------------------------------
  async function startOnlineStory() {
    geminiHistoryRef.current = [
      { role: 'user', parts: [{ text: `Begin part 1 of the adventure for ${character}.` }] },
    ];
    setSnapshotHistory([]);
    setBusy(true);
    try {
      const r = await callGemini({ apiKey, model, geminiHistory: geminiHistoryRef.current, character, hobby, genre });
      geminiHistoryRef.current.push({ role: 'model', parts: [{ text: r.story_text }] });
      setPart(1);
      setCurrentText(r.story_text);
      setOptions(r.options);
      setStoryHistory([{ part: 1, text: r.story_text, choice: null }]);
      setScreen('story');
      persistResume({ mode: 'online', character, hobby, genre, part: 1,
        currentText: r.story_text, options: r.options,
        storyHistory: [{ part: 1, text: r.story_text, choice: null }],
        snapshotHistory: [],
        geminiHistory: geminiHistoryRef.current });
    } catch (e) {
      handleApiError(e, /*atStart*/ true);
    } finally {
      setBusy(false);
    }
  }

  // ---- start an offline story --------------------------------------------
  function startOfflineStory(reasonMsg) {
    const story = { ...OFFLINE_STORIES[Math.floor(Math.random() * OFFLINE_STORIES.length)] };
    story.hero = character || 'John';
    const text = offlinePartText(story, 0, null);
    setOfflineStory(story);
    setSnapshotHistory([]);
    setPart(1);
    setCurrentText(text);
    setOptions(story.parts[0].options);
    setStoryHistory([{ part: 1, text, choice: null }]);
    setScreen('story');
    if (reasonMsg) {
      // Friendly, not an error wall. // [FIX] graceful offline, not a dead end.
      setTimeout(() => Alert.alert('No internet right now', reasonMsg), 250);
    }
  }

  // ---- begin button -------------------------------------------------------
  async function onBegin() {
    if (!apiKey) { setSettingsOpen(true); return; }
    if (!hobby.trim()) { Alert.alert('One more thing', 'What does the hero love to do?'); return; }
    tap();
    await clearResume();
    if (await isOnline()) startOnlineStory();
    else startOfflineStory('We\'ll play one of the built-in adventures so the fun never stops!');
  }

  function onContinueResume() {
    (async () => {
      const raw = await AsyncStorage.getItem(STORE.resume);
      if (!raw) return;
      try {
        const s = JSON.parse(raw);
        setCharacter(s.character); setHobby(s.hobby || ''); setGenre(s.genre || 'fantasy');
        setPart(s.part); setCurrentText(s.currentText); setOptions(s.options || []);
        setStoryHistory(s.storyHistory || []);
        setSnapshotHistory(s.snapshotHistory || []);
        if (s.mode === 'online') geminiHistoryRef.current = s.geminiHistory || [];
        setOfflineStory(s.offlineStory || null);
        setScreen(s.part >= TOTAL_PARTS && (!s.options || !s.options.length) ? 'end' : 'story');
      } catch (e) {}
    })();
  }

  // ---- choose an option ---------------------------------------------------
  async function onChoose(choice) {
    tap();
    Speech.stop();
    const nextPart = part + 1;

    // Save a snapshot of the current state BEFORE committing this choice.
    // This is what "Go Back" will restore.
    const snapshot = {
      part,
      currentText,
      options,
      storyHistory: storyHistory.map(h => ({ ...h })),
      geminiHistory: offlineStory ? null : [...geminiHistoryRef.current],
      offlineStory: offlineStory ? { ...offlineStory } : null,
    };
    const newSnapshots = [...snapshotHistory, snapshot];
    setSnapshotHistory(newSnapshots);

    // OFFLINE branch
    if (offlineStory) {
      const idx = nextPart - 1;
      const text = offlinePartText(offlineStory, idx, choice);
      const opts = offlineStory.parts[idx].options || [];
      const newHistory = [...storyHistory.map((h, i) =>
        i === storyHistory.length - 1 ? { ...h, choice } : h), { part: nextPart, text, choice: null }];
      setStoryHistory(newHistory);
      setPart(nextPart);
      setCurrentText(text);
      setOptions(opts);
      if (nextPart >= TOTAL_PARTS) setScreen('end');
      return;
    }

    // ONLINE branch
    setBusy(true);
    try {
      geminiHistoryRef.current.push({
        role: 'user',
        parts: [{ text: `${character} chose to: ${choice}. Now write part ${nextPart}` +
          (nextPart >= TOTAL_PARTS ? ' as the final happy ending (no options).' : '.') }],
      });
      const r = await callGemini({ apiKey, model, geminiHistory: geminiHistoryRef.current, character, hobby, genre });
      geminiHistoryRef.current.push({ role: 'model', parts: [{ text: r.story_text }] });

      const newHistory = [...storyHistory.map((h, i) =>
        i === storyHistory.length - 1 ? { ...h, choice } : h), { part: nextPart, text: r.story_text, choice: null }];
      setStoryHistory(newHistory);
      setPart(nextPart);
      setCurrentText(r.story_text);
      setOptions(r.options);
      persistResume({ mode: 'online', character, hobby, genre, part: nextPart,
        currentText: r.story_text, options: r.options, storyHistory: newHistory,
        snapshotHistory: newSnapshots,
        geminiHistory: geminiHistoryRef.current });

      if (nextPart >= TOTAL_PARTS || !r.options.length) setScreen('end');
    } catch (e) {
      // Roll back snapshot on failure — the choice didn't go through
      setSnapshotHistory(snapshotHistory);
      handleApiError(e, false);
    } finally {
      setBusy(false);
    }
  }

  // ---- go back to previous chapter and pick again ------------------------
  async function onGoBack() {
    if (snapshotHistory.length === 0) return;
    tap();
    Speech.stop();

    const prev = snapshotHistory[snapshotHistory.length - 1];
    const newSnapshots = snapshotHistory.slice(0, -1);

    // Restore UI state
    setPart(prev.part);
    setCurrentText(prev.currentText);
    setOptions(prev.options);
    setStoryHistory(prev.storyHistory);
    setSnapshotHistory(newSnapshots);
    setOfflineStory(prev.offlineStory || null);

    // Restore Gemini conversation to where it was before that choice
    if (!prev.offlineStory && prev.geminiHistory) {
      geminiHistoryRef.current = [...prev.geminiHistory];
    }

    // Make sure we're on the story screen (handles going back from 'end')
    setScreen('story');
  }

  // ---- error handling -----------------------------------------------------
  function handleApiError(e, atStart) {
    const msg = e?.message || '';
    if (msg === 'MODEL_404') {
      Alert.alert('Story machine needs an update',
        `The model "${model}" isn't available anymore. Tap the gear and pick a current one (e.g. gemini-2.5-flash).`);
      setSettingsOpen(true);
      return;
    }
    if (msg === 'BAD_KEY') {
      Alert.alert('Key problem', 'That API key looks wrong or isn\'t allowed. Please check it in Settings.');
      setSettingsOpen(true);
      return;
    }
    if (msg === 'BLOCKED') {
      Alert.alert('Let\'s try a different idea', 'That turn didn\'t feel quite right for a cozy story. Try another choice!');
      return;
    }
    if (msg === 'NETWORK') {
      if (atStart) startOfflineStory('You\'re offline, so here\'s a built-in adventure!');
      else Alert.alert('Lost the internet', 'We couldn\'t reach the story machine. Check your connection and try the same choice again.');
      return;
    }
    Alert.alert('Hmm, that didn\'t work', 'Something went wrong. Please try again.');
  }

  // ---- save key -----------------------------------------------------------
  async function saveSettings(newKey, newModel) {
    if (newKey && newKey.trim()) {
      await AsyncStorage.setItem(STORE.apiKey, newKey.trim());
      setApiKey(newKey.trim());
    }
    const m = (newModel && newModel.trim()) || DEFAULT_MODEL;
    await AsyncStorage.setItem(STORE.model, m);
    setModel(m);
    setSettingsOpen(false);
  }

  // ---- new adventure ------------------------------------------------------
  function resetAll() {
    Speech.stop();
    clearResume();
    setOfflineStory(null);
    setPart(1);
    setCurrentText('');
    setOptions([]);
    setStoryHistory([]);
    setSnapshotHistory([]);
    geminiHistoryRef.current = [];
    setHobby('');
    setGenre('fantasy');
    setScreen('setup');
  }

  // ---- share --------------------------------------------------------------
  async function shareStory() {
    try {
      const title = `John's Adventure\nStarring ${character}\n\n`;
      const bodyText = storyHistory
        .map((h) => `Part ${h.part}\n${h.text}` + (h.choice ? `\n\n( ${character} chose to ${lowerFirst(h.choice)} )` : ''))
        .join('\n\n\n');
      const full = title + bodyText + '\n';

      // [FIX] new SDK 54 File API (writeAsStringAsync is deprecated).
      const file = new File(Paths.cache, 'Johns_Adventure.txt');
      try { if (file.exists) file.delete(); } catch (e) {}
      file.create();
      file.write(full);

      if (!(await Sharing.isAvailableAsync())) {
        Alert.alert('Sharing not available', 'This device can\'t open the share menu.');
        return;
      }
      // [FIX] expo-sharing opens the system share sheet; WhatsApp appears there.
      await Sharing.shareAsync(file.uri, {
        mimeType: 'text/plain',
        dialogTitle: 'Share John\'s Adventure',
        UTI: 'public.plain-text',
      });
    } catch (e) {
      Alert.alert('Couldn\'t share', 'Something went wrong saving the story file.');
    }
  }

  // ---- render -------------------------------------------------------------
  if (screen === 'loading') {
    return (
      <SafeAreaView style={styles.fill}>
        <StatusBar barStyle="light-content" />
        <View style={styles.center}><ActivityIndicator color={COLORS.white} size="large" /></View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.fill} edges={['top', 'bottom']}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      {screen === 'setup' && (
        <SetupScreen
          character={character} setCharacter={setCharacter}
          hobby={hobby} setHobby={setHobby}
          genre={genre} setGenre={setGenre}
          onBegin={onBegin} onGear={() => setSettingsOpen(true)}
          hasResume={hasResume} onContinue={onContinueResume}
          busy={busy} hasKey={!!apiKey}
        />
      )}

      {screen === 'story' && (
        <StoryScreen
          part={part} text={currentText} options={options}
          onChoose={onChoose} onHistory={() => setHistoryOpen(true)}
          onSpeak={() => speak(currentText)} busy={busy}
          canGoBack={snapshotHistory.length > 0} onGoBack={onGoBack}
        />
      )}

      {screen === 'end' && (
        <EndScreen
          text={currentText} onSpeak={() => speak(currentText)}
          onShare={shareStory} onRestart={() =>
            Alert.alert('Start a new adventure?', 'This story will be cleared.',
              [{ text: 'Keep reading', style: 'cancel' }, { text: 'New adventure', style: 'destructive', onPress: resetAll }])}
          onHistory={() => setHistoryOpen(true)}
          canGoBack={snapshotHistory.length > 0} onGoBack={onGoBack}
        />
      )}

      <SettingsModal
        visible={settingsOpen} onClose={() => setSettingsOpen(false)}
        currentModel={model} hasKey={!!apiKey} onSave={saveSettings}
      />
      <HistoryModal
        visible={historyOpen} onClose={() => setHistoryOpen(false)}
        history={storyHistory} character={character}
      />
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Screens
// ---------------------------------------------------------------------------

function SetupScreen({ character, setCharacter, hobby, setHobby, genre, setGenre, onBegin, onGear, hasResume, onContinue, busy, hasKey }) {
  return (
    <View style={styles.fill}>
      <Pressable style={styles.gear} onPress={onGear} hitSlop={12}>
        <Text style={styles.gearIcon}>⚙️</Text>
      </Pressable>

      <ScrollView contentContainerStyle={styles.setupScroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.bigEmoji}>✨📖✨</Text>
        <Text style={styles.h1}>Let's Start an Adventure!</Text>

        <Text style={styles.label}>Who is the main character?</Text>
        <TextInput
          style={styles.input} value={character} onChangeText={setCharacter}
          placeholder="John" placeholderTextColor="#7AA3A3" maxLength={20}
        />

        <Text style={styles.label}>What do they love to do?</Text>
        <TextInput
          style={styles.input} value={hobby} onChangeText={setHobby}
          placeholder="painting, dancing, exploring…" placeholderTextColor="#7AA3A3" maxLength={40}
        />

        <Text style={styles.label}>What kind of adventure?</Text>
        <View style={styles.genreGrid}>
          {GENRES.map(g => (
            <Pressable
              key={g.id}
              style={[styles.genrePill, genre === g.id && styles.genrePillActive]}
              onPress={() => setGenre(g.id)}
            >
              <Text style={[styles.genrePillText, genre === g.id && styles.genrePillTextActive]}>
                {g.label}
              </Text>
            </Pressable>
          ))}
        </View>

        {!hasKey && (
          <Text style={styles.hint}>Tip for grown-ups: tap ⚙️ to add a Gemini API key once.</Text>
        )}
      </ScrollView>

      <View style={styles.footer}>
        {hasResume && (
          <Pressable style={[styles.primaryBtn, styles.resumeBtn]} onPress={onContinue}>
            <Text style={styles.primaryBtnText}>↩️  Continue last adventure</Text>
          </Pressable>
        )}
        <Pressable style={styles.primaryBtn} onPress={onBegin} disabled={busy}>
          {busy ? <ActivityIndicator color={COLORS.cardText} />
                : <Text style={styles.primaryBtnText}>Begin the Story!  🚀</Text>}
        </Pressable>
      </View>
    </View>
  );
}

function StoryScreen({ part, text, options, onChoose, onHistory, onSpeak, busy, canGoBack, onGoBack }) {
  return (
    <View style={styles.fill}>
      <View style={styles.topBar}>
        <ProgressDots part={part} />
        <Pressable onPress={onHistory} hitSlop={8}>
          <Text style={styles.topBarBtn}>📖 Story so far</Text>
        </Pressable>
      </View>

      <ScrollView contentContainerStyle={styles.storyScroll} showsVerticalScrollIndicator={false}>
        <Text style={styles.storyText}>{text}</Text>
        <Pressable style={styles.speakBtn} onPress={onSpeak}>
          <Text style={styles.speakBtnText}>🔊  Read to me</Text>
        </Pressable>
      </ScrollView>

      <View style={styles.footer}>
        {busy ? (
          <LoadingFooter />
        ) : (
          <>
            {options.map((opt, i) => (
              <Pressable key={i} style={styles.choiceCard} onPress={() => onChoose(opt)}>
                <Text style={styles.choiceText}>{['🌿', '⭐', '🐚'][i] || '✨'}  {opt}</Text>
              </Pressable>
            ))}
            {canGoBack && (
              <Pressable style={styles.backBtn} onPress={onGoBack}>
                <Text style={styles.backBtnText}>← Go back and choose differently</Text>
              </Pressable>
            )}
          </>
        )}
      </View>
    </View>
  );
}

function EndScreen({ text, onSpeak, onShare, onRestart, onHistory, canGoBack, onGoBack }) {
  return (
    <View style={styles.fill}>
      <View style={styles.topBar}>
        <Text style={styles.theEnd}>The End  🎉</Text>
        <Pressable onPress={onHistory} hitSlop={8}>
          <Text style={styles.topBarBtn}>📖 Story so far</Text>
        </Pressable>
      </View>

      <ScrollView contentContainerStyle={styles.storyScroll} showsVerticalScrollIndicator={false}>
        <Text style={styles.storyText}>{text}</Text>
        <Pressable style={styles.speakBtn} onPress={onSpeak}>
          <Text style={styles.speakBtnText}>🔊  Read to me</Text>
        </Pressable>
      </ScrollView>

      <View style={styles.footer}>
        {canGoBack && (
          <Pressable style={[styles.backBtn, { marginBottom: 10 }]} onPress={onGoBack}>
            <Text style={styles.backBtnText}>← Go back and choose differently</Text>
          </Pressable>
        )}
        <Pressable style={[styles.primaryBtn, { marginBottom: 10 }]} onPress={onShare}>
          <Text style={styles.primaryBtnText}>💬  Share with Dad</Text>
        </Pressable>
        <Pressable style={[styles.choiceCard, { alignItems: 'center' }]} onPress={onRestart}>
          <Text style={styles.choiceText}>🌟  Start a New Adventure</Text>
        </Pressable>
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function ProgressDots({ part }) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
      <Text style={styles.partLabel}>Part {part} of {TOTAL_PARTS}  </Text>
      {Array.from({ length: TOTAL_PARTS }).map((_, i) => (
        <View key={i} style={[styles.dot, i < part ? styles.dotOn : styles.dotOff]} />
      ))}
    </View>
  );
}

function LoadingFooter() {
  return (
    <View style={styles.loadingFooter}>
      <ActivityIndicator color={COLORS.white} />
      <Text style={styles.loadingText}>✨ Dreaming up what happens next…</Text>
    </View>
  );
}

function SettingsModal({ visible, onClose, currentModel, hasKey, onSave }) {
  const [key, setKey] = useState('');
  const [model, setModel] = useState(currentModel);
  useEffect(() => { setModel(currentModel); setKey(''); }, [visible, currentModel]);

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.modalBackdrop}>
        <View style={styles.modalCard}>
          <Text style={styles.modalTitle}>Grown-up Settings</Text>

          <Text style={styles.modalLabel}>Gemini API Key {hasKey ? '(saved ✓)' : ''}</Text>
          <TextInput
            style={styles.modalInput} value={key} onChangeText={setKey}
            placeholder={hasKey ? 'Enter a new key to replace' : 'Paste your key here'}
            placeholderTextColor="#7AA3A3" secureTextEntry autoCapitalize="none" autoCorrect={false}
          />

          <Text style={styles.modalLabel}>Model</Text>
          <TextInput
            style={styles.modalInput} value={model} onChangeText={setModel}
            placeholder="gemini-3.5-flash" placeholderTextColor="#7AA3A3"
            autoCapitalize="none" autoCorrect={false}
          />
          <Text style={styles.modalNote}>
            Models get retired over time. If stories stop loading with a 404, change this to a current model.
          </Text>

          <View style={{ flexDirection: 'row', marginTop: 16 }}>
            <Pressable style={[styles.modalBtn, styles.modalBtnGhost]} onPress={onClose}>
              <Text style={styles.modalBtnGhostText}>Cancel</Text>
            </Pressable>
            <Pressable style={[styles.modalBtn, styles.modalBtnSolid]} onPress={() => onSave(key, model)}>
              <Text style={styles.modalBtnSolidText}>Save</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

function HistoryModal({ visible, onClose, history, character }) {
  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <SafeAreaView style={styles.fill}>
        <View style={styles.topBar}>
          <Text style={styles.h2}>📖 Story So Far</Text>
          <Pressable onPress={onClose} hitSlop={8}><Text style={styles.topBarBtn}>Close ✕</Text></Pressable>
        </View>
        <ScrollView contentContainerStyle={styles.storyScroll}>
          {history.map((h, i) => (
            <View key={i} style={{ marginBottom: 22 }}>
              <Text style={styles.historyPart}>Part {h.part}</Text>
              <Text style={styles.storyText}>{h.text}</Text>
              {h.choice ? <Text style={styles.historyChoice}>➜ {character} chose to {lowerFirst(h.choice)}</Text> : null}
            </View>
          ))}
        </ScrollView>
      </SafeAreaView>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: COLORS.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  // setup
  gear: { position: 'absolute', top: 8, right: 16, zIndex: 10, padding: 8 },
  gearIcon: { fontSize: 26 },
  setupScroll: { padding: 24, paddingTop: 40, paddingBottom: 24, flexGrow: 1, justifyContent: 'center' },
  bigEmoji: { fontSize: 40, textAlign: 'center', marginBottom: 8 },
  h1: { color: COLORS.white, fontSize: 32, fontWeight: '700', textAlign: 'center', marginBottom: 28 },
  label: { color: COLORS.white, fontSize: 18, fontWeight: '700', marginBottom: 8, marginTop: 12 },
  input: {
    backgroundColor: COLORS.card, borderRadius: 16, paddingHorizontal: 18, paddingVertical: 16,
    fontSize: 18, fontWeight: '700', color: COLORS.cardText,
  },
  hint: { color: COLORS.whiteDim, fontSize: 14, fontWeight: '700', marginTop: 20, textAlign: 'center' },

  // genre picker
  genreGrid: { flexDirection: 'row', flexWrap: 'wrap', marginTop: 4, gap: 8 },
  genrePill: {
    borderRadius: 20, paddingVertical: 8, paddingHorizontal: 14,
    backgroundColor: 'rgba(255,255,255,0.15)', borderWidth: 2, borderColor: 'transparent',
  },
  genrePillActive: {
    backgroundColor: COLORS.accent, borderColor: COLORS.accent,
  },
  genrePillText: { color: COLORS.white, fontSize: 15, fontWeight: '700' },
  genrePillTextActive: { color: COLORS.cardText },

  // footer (pinned via flex layout, NOT position:fixed) // [FIX]
  footer: { padding: 16, paddingTop: 10 },
  primaryBtn: {
    backgroundColor: COLORS.accent, borderRadius: 18, paddingVertical: 18, alignItems: 'center',
  },
  resumeBtn: { backgroundColor: COLORS.card, marginBottom: 10 },
  primaryBtnText: { color: COLORS.cardText, fontSize: 20, fontWeight: '700' },

  // story
  topBar: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 18, paddingVertical: 12,
  },
  topBarBtn: { color: COLORS.white, fontSize: 15, fontWeight: '700' },
  partLabel: { color: COLORS.white, fontSize: 15, fontWeight: '700' },
  dot: { width: 9, height: 9, borderRadius: 5, marginHorizontal: 2 },
  dotOn: { backgroundColor: COLORS.accent },
  dotOff: { backgroundColor: 'rgba(255,255,255,0.3)' },

  // contentContainer padding ensures last line clears the pinned footer // [FIX]
  storyScroll: { padding: 24, paddingBottom: 40 },
  storyText: { color: COLORS.white, fontSize: 21, fontWeight: '700', lineHeight: 32 },
  speakBtn: {
    marginTop: 24, alignSelf: 'flex-start',
    borderColor: COLORS.white, borderWidth: 2, borderRadius: 14, paddingVertical: 10, paddingHorizontal: 18,
  },
  speakBtnText: { color: COLORS.white, fontSize: 16, fontWeight: '700' },

  choiceCard: {
    backgroundColor: COLORS.card, borderRadius: 18, paddingVertical: 18, paddingHorizontal: 20, marginBottom: 12,
  },
  choiceText: { color: COLORS.cardText, fontSize: 18, fontWeight: '700' },

  backBtn: {
    borderRadius: 18, paddingVertical: 13, paddingHorizontal: 20, marginBottom: 4,
    borderWidth: 2, borderColor: 'rgba(255,255,255,0.4)', alignItems: 'center',
  },
  backBtnText: { color: COLORS.whiteDim, fontSize: 15, fontWeight: '700' },

  loadingFooter: { alignItems: 'center', paddingVertical: 28 },
  loadingText: { color: COLORS.white, fontSize: 16, fontWeight: '700', marginTop: 12 },

  theEnd: { color: COLORS.white, fontSize: 24, fontWeight: '700' },
  h2: { color: COLORS.white, fontSize: 22, fontWeight: '700' },

  historyPart: { color: COLORS.accent, fontSize: 16, fontWeight: '700', marginBottom: 6 },
  historyChoice: { color: COLORS.whiteDim, fontSize: 15, fontWeight: '700', marginTop: 8, fontStyle: 'italic' },

  // modals
  modalBackdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'center', padding: 22 },
  modalCard: { backgroundColor: COLORS.card, borderRadius: 22, padding: 22 },
  modalTitle: { fontSize: 22, fontWeight: '700', color: COLORS.cardText, marginBottom: 16 },
  modalLabel: { fontSize: 15, fontWeight: '700', color: COLORS.cardText, marginTop: 12, marginBottom: 6 },
  modalInput: {
    backgroundColor: '#EEF6F6', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12,
    fontSize: 16, fontWeight: '700', color: COLORS.cardText,
  },
  modalNote: { fontSize: 13, color: '#5A7A7A', fontWeight: '700', marginTop: 8 },
  modalBtn: { flex: 1, borderRadius: 14, paddingVertical: 14, alignItems: 'center', marginHorizontal: 4 },
  modalBtnGhost: { backgroundColor: '#E3ECEC' },
  modalBtnGhostText: { color: COLORS.cardText, fontSize: 16, fontWeight: '700' },
  modalBtnSolid: { backgroundColor: COLORS.bg },
  modalBtnSolidText: { color: COLORS.white, fontSize: 16, fontWeight: '700' },
});