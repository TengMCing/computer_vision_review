<script src="https://cdn.jsdelivr.net/npm/mermaid@11.6.0/dist/mermaid.min.js"></script>
<script> 
function refreshTime() {
  var timeDisplay = document.getElementById("mel-local-time");
  var timeString = new Date().toLocaleTimeString("en-US", {timeZone: "Australia/Melbourne"});
  timeDisplay.innerHTML = timeString;
}

setInterval(refreshTime, 1000);
</script>