import { describe, it, expect } from "vitest";
import { alignSpeakers } from "./aligner.js";
import type { WhisperResult, DiarizationSegment } from "./types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeWord(word: string, start: number, end: number) {
  return { word, start, end };
}

function makeSegment(text: string, start: number, end: number, words: ReturnType<typeof makeWord>[]) {
  return { text, start, end, words };
}

function makeWhisperResult(segments: ReturnType<typeof makeSegment>[]): WhisperResult {
  return { segments, language: "es" };
}

function makeSpeaker(speaker: string, start: number, end: number): DiarizationSegment {
  return { speaker, start, end };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("alignSpeakers", () => {
  // 1. Empty whisperResult → returns []
  it("returns [] when whisperResult has no segments", () => {
    const result = alignSpeakers(makeWhisperResult([]), []);
    expect(result).toEqual([]);
  });

  it("returns [] when whisperResult has no segments even with speakers", () => {
    const speakers = [makeSpeaker("SPEAKER_00", 0, 10)];
    const result = alignSpeakers(makeWhisperResult([]), speakers);
    expect(result).toEqual([]);
  });

  // 2. Empty speakers array → returns segments with speaker: null
  it("returns AttributedSegments with speaker=null when speakers array is empty", () => {
    const whisper = makeWhisperResult([
      makeSegment("Hola mundo", 0, 3, [makeWord("Hola", 0, 1), makeWord("mundo", 1, 3)]),
      makeSegment("Como estas", 4, 7, [makeWord("Como", 4, 5), makeWord("estas", 5, 7)]),
    ]);

    const result = alignSpeakers(whisper, []);
    expect(result).toHaveLength(2);
    expect(result[0].speaker).toBeNull();
    expect(result[0].text).toBe("Hola mundo");
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(3);
    expect(result[1].speaker).toBeNull();
    expect(result[1].text).toBe("Como estas");
  });

  // 3. Single speaker, all words aligned → one AttributedSegment per run
  it("produces one AttributedSegment when all words belong to the same speaker", () => {
    const whisper = makeWhisperResult([
      makeSegment("Hola como estas", 0, 6, [
        makeWord("Hola", 0, 2),
        makeWord(" como", 2, 4),
        makeWord(" estas", 4, 6),
      ]),
    ]);
    const speakers = [makeSpeaker("SPEAKER_00", 0, 6)];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBe("SPEAKER_00");
    expect(result[0].text).toBe("Hola como estas");
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(6);
  });

  // 4. Two speakers alternating → multiple AttributedSegments grouped by speaker
  it("splits words into multiple segments when two speakers alternate", () => {
    const whisper = makeWhisperResult([
      makeSegment("Buenos dias hola", 0, 6, [
        makeWord("Buenos", 0, 2),
        makeWord(" dias", 2, 4),
        makeWord(" hola", 10, 12),
      ]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 4),
      makeSpeaker("SPEAKER_01", 10, 14),
    ];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(2);
    expect(result[0].speaker).toBe("SPEAKER_00");
    expect(result[0].text).toContain("Buenos");
    expect(result[1].speaker).toBe("SPEAKER_01");
    expect(result[1].text).toContain("hola");
  });

  // 5. Word falls in gap between speakers → assigned to nearest by temporal distance
  it("assigns a word in a gap to the nearest speaker by temporal distance", () => {
    // Gap: speaker 0 ends at 3.0, speaker 1 starts at 7.0; gap is 3.0..7.0
    // Word at 3.5 is 0.5s from speaker 0 end, 3.5s from speaker 1 start → speaker 0
    const whisper = makeWhisperResult([
      makeSegment("Prueba", 3.5, 4.0, [makeWord("Prueba", 3.5, 4.0)]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 3.0),
      makeSpeaker("SPEAKER_01", 7.0, 10.0),
    ];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBe("SPEAKER_00");
  });

  it("assigns a word closer to the second speaker when it is nearer", () => {
    // Gap: speaker 0 ends at 3.0, speaker 1 starts at 7.0
    // Word at 6.6 is 3.6s from speaker 0 end, 0.4s from speaker 1 start → speaker 1
    const whisper = makeWhisperResult([
      makeSegment("Oye", 6.6, 7.0, [makeWord("Oye", 6.6, 7.0)]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 3.0),
      makeSpeaker("SPEAKER_01", 7.0, 10.0),
    ];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBe("SPEAKER_01");
  });

  // 6. 500ms boundary tolerance — word at speaker.end + 0.3s should still match that speaker
  it("matches a word 0.3s beyond speaker end due to 500ms tolerance", () => {
    // speaker 0: 0..5. Word at 5.3 (300ms past end). With 500ms tolerance → strictly within window
    const whisper = makeWhisperResult([
      makeSegment("Listo", 5.3, 5.8, [makeWord("Listo", 5.3, 5.8)]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 5.0),
      makeSpeaker("SPEAKER_01", 20.0, 25.0),
    ];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBe("SPEAKER_00");
  });

  it("does NOT match a word 0.6s beyond speaker end when tolerance is default 500ms", () => {
    // speaker 0: 0..5. Word at 5.6. With 500ms tolerance speaker0 window extends to 5.5 only.
    // speaker 1 starts at 20.0. Nearest by distance: speaker0 is 0.6s away, speaker1 is 14.4s away.
    // Still assigned to speaker0 because it is the nearest overall (fallback path).
    const whisper = makeWhisperResult([
      makeSegment("Bueno", 5.6, 6.0, [makeWord("Bueno", 5.6, 6.0)]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 5.0),
      makeSpeaker("SPEAKER_01", 20.0, 25.0),
    ];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    // nearest is still SPEAKER_00 (0.6s away vs 14.4s away)
    expect(result[0].speaker).toBe("SPEAKER_00");
  });

  // 7. Multiple consecutive words from same speaker → grouped into single paragraph
  it("groups multiple consecutive same-speaker words into one paragraph", () => {
    const whisper = makeWhisperResult([
      makeSegment("La reunion empieza ahora mismo gracias", 0, 10, [
        makeWord("La", 0, 1),
        makeWord(" reunion", 1, 2),
        makeWord(" empieza", 2, 4),
        makeWord(" ahora", 4, 6),
        makeWord(" mismo", 6, 8),
        makeWord(" gracias", 8, 10),
      ]),
    ]);
    const speakers = [makeSpeaker("SPEAKER_00", 0, 10)];

    const result = alignSpeakers(whisper, speakers);
    // All 6 words should be merged into ONE segment
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBe("SPEAKER_00");
    expect(result[0].text).toBe("La reunion empieza ahora mismo gracias");
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(10);
  });

  it("aggregates text correctly across multiple segments with same speaker", () => {
    const whisper = makeWhisperResult([
      makeSegment("Hola", 0, 2, [makeWord("Hola", 0, 2)]),
      makeSegment(" mundo", 2, 4, [makeWord(" mundo", 2, 4)]),
    ]);
    const speakers = [makeSpeaker("SPEAKER_00", 0, 5)];

    const result = alignSpeakers(whisper, speakers);
    expect(result).toHaveLength(1);
    expect(result[0].text).toBe("Hola mundo");
  });

  // Edge case: segments with no word-level tokens fall back gracefully
  it("degrades gracefully when segments have no words (empty words array)", () => {
    const whisper = makeWhisperResult([
      makeSegment("Texto sin palabras", 0, 5, []),
    ]);
    const speakers = [makeSpeaker("SPEAKER_00", 0, 5)];

    const result = alignSpeakers(whisper, speakers);
    // words.length === 0 → graceful degradation with speaker=null
    expect(result).toHaveLength(1);
    expect(result[0].speaker).toBeNull();
    expect(result[0].text).toBe("Texto sin palabras");
  });

  // Custom tolerance
  it("respects custom toleranceMs parameter", () => {
    // With 0ms tolerance, a word 100ms outside boundary won't be in window
    // but is still assigned to the nearest speaker by fallback
    const whisper = makeWhisperResult([
      makeSegment("Test", 5.1, 5.5, [makeWord("Test", 5.1, 5.5)]),
    ]);
    const speakers = [
      makeSpeaker("SPEAKER_00", 0, 5.0),
      makeSpeaker("SPEAKER_01", 20.0, 25.0),
    ];

    const result = alignSpeakers(whisper, speakers, 0);
    expect(result).toHaveLength(1);
    // SPEAKER_00 is 0.1s away; SPEAKER_01 is 14.9s away
    expect(result[0].speaker).toBe("SPEAKER_00");
  });
});
