import type { WhisperResult, WhisperWord, DiarizationSegment, AttributedSegment } from "./types.js";

/**
 * Find the index of the diarization segment that best matches a word's start time.
 *
 * Strategy:
 *  1. Binary-search for a segment where word.start falls within
 *     [speaker.start - toleranceSec, speaker.end + toleranceSec].
 *  2. If multiple candidates match, prefer the one that strictly contains word.start.
 *  3. If no candidate matches the tolerance window, pick the temporally nearest segment.
 */
function findSpeakerIndex(
  wordStart: number,
  speakers: DiarizationSegment[],
  toleranceSec: number
): number {
  // Binary search: find the leftmost segment whose end + tolerance >= wordStart
  let lo = 0;
  let hi = speakers.length - 1;
  let candidateIdx = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const seg = speakers[mid];

    if (seg.end + toleranceSec >= wordStart) {
      candidateIdx = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  // Scan forward from candidateIdx to collect all segments within the tolerance window
  let bestIdx = -1;
  let bestDist = Infinity;

  const start = candidateIdx === -1 ? 0 : candidateIdx;
  for (let i = start; i < speakers.length; i++) {
    const seg = speakers[i];
    if (seg.start - toleranceSec > wordStart) break; // past the window

    // Strict containment wins
    if (wordStart >= seg.start && wordStart <= seg.end) {
      return i;
    }

    // Otherwise track nearest by distance to [start, end] interval
    const dist = Math.min(
      Math.abs(wordStart - seg.start),
      Math.abs(wordStart - seg.end)
    );
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }

  if (bestIdx !== -1) return bestIdx;

  // Fall back: nearest segment overall (should rarely happen)
  let nearestIdx = 0;
  let nearestDist = Infinity;
  for (let i = 0; i < speakers.length; i++) {
    const seg = speakers[i];
    const dist = Math.min(
      Math.abs(wordStart - seg.start),
      Math.abs(wordStart - seg.end)
    );
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestIdx = i;
    }
  }
  return nearestIdx;
}

/**
 * Align word-level Whisper output with pyannote speaker diarization segments.
 *
 * @param whisperResult  Full Whisper output including per-segment word tokens.
 * @param speakers       Sorted diarization segments (speaker + time range).
 * @param toleranceMs    How many milliseconds to expand each diarization boundary
 *                       when searching for a matching speaker. Default 500 ms.
 * @returns              AttributedSegment[] — consecutive words with the same
 *                       speaker are merged into a single paragraph.
 */
export function alignSpeakers(
  whisperResult: WhisperResult,
  speakers: DiarizationSegment[],
  toleranceMs: number = 500
): AttributedSegment[] {
  if (whisperResult.segments.length === 0) {
    return [];
  }

  // Flatten all words from all segments
  const words: WhisperWord[] = whisperResult.segments.flatMap((seg) => seg.words);

  // Graceful degradation: no diarization data → single speaker=null per segment
  if (speakers.length === 0) {
    return whisperResult.segments.map((seg) => ({
      speaker: null,
      text: seg.text.trim(),
      start: seg.start,
      end: seg.end,
    }));
  }

  if (words.length === 0) {
    // Segments exist but no word-level tokens: degrade gracefully
    return whisperResult.segments.map((seg) => ({
      speaker: null,
      text: seg.text.trim(),
      start: seg.start,
      end: seg.end,
    }));
  }

  const toleranceSec = toleranceMs / 1000;

  // Assign each word to a speaker
  const wordSpeakers: string[] = words.map((word) => {
    const idx = findSpeakerIndex(word.start, speakers, toleranceSec);
    return speakers[idx].speaker;
  });

  // Group consecutive words with the same speaker into AttributedSegment paragraphs
  const result: AttributedSegment[] = [];
  let groupStart = 0;

  for (let i = 1; i <= words.length; i++) {
    const speakerChanged = i === words.length || wordSpeakers[i] !== wordSpeakers[groupStart];

    if (speakerChanged) {
      const groupWords = words.slice(groupStart, i);
      const text = groupWords.map((w) => w.word).join("").trim();
      result.push({
        speaker: wordSpeakers[groupStart],
        text,
        start: groupWords[0].start,
        end: groupWords[groupWords.length - 1].end,
      });
      groupStart = i;
    }
  }

  return result;
}
