let uploadId = null;

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a video file');
        return;
    }
    
    // Show progress section
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    try {
        // Upload file
        updateProgress(10, 'Uploading video...');
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const uploadData = await uploadResponse.json();
        uploadId = uploadData.upload_id;
        
        // Process video
        updateProgress(20, 'Starting AI processing...');
        
        const numClips = document.getElementById('numClips').value;
        const clipDuration = document.getElementById('clipDuration').value;
        const minGap = document.getElementById('minGap').value;
        
        const processResponse = await fetch(
            `/api/process/${uploadId}?num_clips=${numClips}&clip_duration=${clipDuration}&min_gap=${minGap}&use_ai=true`,
            { method: 'POST' }
        );
        
        const processData = await processResponse.json();
        
        if (processData.success) {
            updateProgress(100, 'Processing complete!');
            displayResults(processData);
        } else {
            throw new Error('Processing failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing video: ' + error.message);
        resetForm();
    }
});

function updateProgress(percent, message) {
    document.getElementById('progressBar').value = percent;
    document.getElementById('progressPercent').textContent = percent;
    document.getElementById('progressMessage').textContent = message;
}

function displayResults(data) {
    // Hide progress, show results
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    // Success banner
    document.getElementById('successBanner').style.display = 'block';
    document.getElementById('clipCount').textContent = data.num_clips;
    
    // Analytics
    const analyticsContainer = document.getElementById('analyticsContainer');
    analyticsContainer.innerHTML = '';
    
    data.clips.forEach((clip, index) => {
        const confidence = Math.round(clip.confidence_score * 100);
        const peakTime = formatTime(clip.peak_time);
        const keyword = clip.keyword || 'High-energy moment';
        
        const statBox = document.createElement('div');
        statBox.className = 'stat-box';
        statBox.innerHTML = `
            <h4>Reel ${clip.id}</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Peak at ${peakTime}</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">AI Score: ${confidence}%</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.9;">"${keyword.substring(0, 30)}..."</p>
        `;
        analyticsContainer.appendChild(statBox);
    });
    
    // Download All button
    const downloadAllBtn = document.getElementById('downloadAllBtn');
    downloadAllBtn.classList.remove('hidden');
    downloadAllBtn.onclick = () => {
        window.location.href = data.zip_download;
    };
    
    // Individual clips
    const clipsContainer = document.getElementById('clipsContainer');
    clipsContainer.innerHTML = '';
    
    data.clips.forEach((clip) => {
        const clipCard = createClipCard(clip);
        clipsContainer.appendChild(clipCard);
    });
}

function createClipCard(clip) {
    const confidence = Math.round(clip.confidence_score * 100);
    const peakTime = formatTime(clip.peak_time);
    const fileSize = (clip.file_size / (1024 * 1024)).toFixed(1);
    const keyword = clip.keyword || 'High-energy moment';
    const filename = clip.path.split('/').pop().split('\\').pop();
    
    const card = document.createElement('div');
    card.className = 'clip-card';
    card.innerHTML = `
        <h4>🎬 Reel ${clip.id}: ${keyword.substring(0, 50)}</h4>
        <p>
            <strong>Timestamp:</strong> ${peakTime} | 
            <strong>Duration:</strong> ${clip.duration}s | 
            <strong>AI Confidence:</strong> ${confidence}% | 
            <strong>Size:</strong> ${fileSize} MB
        </p>
        <p><strong>🤖 AI Detection:</strong> Multimodal analysis (Audio + Text Sentiment)</p>
        
        <div class="grid">
            <div>
                <video controls>
                    <source src="/output/${filename}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div>
                ${clip.thumbnail ? `<img src="/output/${clip.thumbnail.split('/').pop().split('\\').pop()}" class="thumbnail" alt="Thumbnail">` : ''}
                <button onclick="window.location.href='/api/download/${filename}'">
                    ⬇️ Download Reel ${clip.id}
                </button>
                <div style="background: #e7f3ff; padding: 0.5rem; border-radius: 0.5rem; margin-top: 0.5rem; font-size: 0.85rem;">
                    <strong>📝 AI Caption:</strong><br>
                    "${keyword.substring(0, 40)}... Watch till the end! #viral #shorts"
                </div>
            </div>
        </div>
    `;
    
    return card;
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function resetForm() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadForm').reset();
}
