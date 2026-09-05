"use strict";
const $ = id => document.getElementById(id);
let item, annotator, frameIndex = 0, fullPlayed = false, playing = false;
let plays = 0, replays = 0, steps = 0, preloadFailures = 0;
let startedAt = 0, hiddenAt = null, hiddenMs = 0, visibilityChanges = 0, events = [];

function event(type, extra={}) {
  events.push({type, t_ms: Math.round(performance.now() - startedAt), ...extra});
}
function frameUrl(index) {
  return `/api/frame/${item.item_id}/${index}?annotator=${encodeURIComponent(annotator)}&assignment=${item.assignment_id}`;
}
function showFrame(index) {
  return new Promise((resolve, reject) => {
    $("frame").onload = () => { frameIndex=index; $("counter").textContent=`Frame ${index+1} / ${item.frame_count}`; resolve(); };
    $("frame").onerror = () => { preloadFailures++; event("preload_failure",{index}); reject(new Error("Frame failed to load")); };
    $("frame").src = frameUrl(index) + `&nonce=${Date.now()}`;
  });
}
async function playFull() {
  if (playing) return;
  playing=true; $("play").disabled=true; $("error").textContent="";
  plays++; if (plays > 1) replays++; event("play_start",{play:plays});
  $("status").textContent="Playing at 16 FPS. Keep this tab visible.";
  try {
    const target=performance.now();
    for (let i=0; i<item.frame_count; i++) {
      if (document.hidden) throw new Error("Playback interrupted because the tab was hidden. Replay in full.");
      await showFrame(i);
      const wait = target + (i+1)*1000/item.fps - performance.now();
      if (wait > 0) await new Promise(r=>setTimeout(r,wait));
    }
    fullPlayed=true; event("play_complete",{play:plays});
    $("status").textContent="Full playback complete. You may replay or step to the exact onset frame.";
    $("prev").disabled=false; $("next").disabled=false; $("response").disabled=false;
    $("confidence").disabled=false; $("submit").disabled=false;
  } catch(e) { fullPlayed=false; $("status").textContent=e.message; event("play_abort",{reason:e.message}); }
  finally { playing=false; $("play").disabled=false; }
}
async function loadNext() {
  const response=await fetch(`/api/next?annotator=${encodeURIComponent(annotator)}`);
  const value=await response.json();
  if (!response.ok) throw new Error(value.error || "Could not obtain item");
  if (value.done) { $("task").innerHTML="<h2>All assigned items are complete.</h2>"; return; }
  item=value; frameIndex=0; fullPlayed=false; plays=0; replays=0; steps=0; preloadFailures=0;
  hiddenMs=0; visibilityChanges=0; events=[]; startedAt=performance.now();
  $("command").textContent=item.command_label;
  $("response").disabled=true; $("confidence").disabled=true; $("submit").disabled=true;
  document.querySelectorAll("[name=response]").forEach(x=>x.checked=false);
  $("confidence").value=""; $("prev").disabled=true; $("next").disabled=true;
  $("status").textContent="Play the entire clip before answering.";
  await showFrame(0); event("item_loaded");
}
$("start").onclick=async()=>{
  annotator=$("annotator").value.trim();
  if(!annotator) return;
  $("login").hidden=true; $("task").hidden=false;
  try { await loadNext(); } catch(e) { $("error").textContent=e.message; }
};
$("play").onclick=playFull;
$("prev").onclick=async()=>{ if(fullPlayed && frameIndex>0){steps++;event("step",{direction:-1});await showFrame(frameIndex-1);} };
$("next").onclick=async()=>{ if(fullPlayed && frameIndex+1<item.frame_count){steps++;event("step",{direction:1});await showFrame(frameIndex+1);} };
document.addEventListener("visibilitychange",()=>{
  visibilityChanges++; event("visibility",{hidden:document.hidden});
  if(document.hidden) hiddenAt=performance.now();
  else if(hiddenAt!==null){hiddenMs+=performance.now()-hiddenAt;hiddenAt=null;}
});
$("submit").onclick=async()=>{
  $("error").textContent="";
  const selected=document.querySelector("[name=response]:checked");
  const confidence=Number($("confidence").value);
  if(!fullPlayed || !selected || !confidence){$("error").textContent="Complete playback and choose a response and confidence.";return;}
  event("submit");
  const payload={
    annotator, assignment_id:item.assignment_id, response:selected.value,
    onset_index:selected.value==="onset"?frameIndex:null, confidence, plays,replays,steps,
    decision_ms:Math.round(performance.now()-startedAt), hidden_ms:Math.round(hiddenMs),
    visibility_changes:visibilityChanges, preload_failures:preloadFailures, events
  };
  $("submit").disabled=true;
  try {
    const response=await fetch("/api/submit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const value=await response.json();
    if(!response.ok) throw new Error(value.error || "Submission failed");
    await loadNext();
  } catch(e) { $("error").textContent=e.message; $("submit").disabled=false; }
};
