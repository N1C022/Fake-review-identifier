document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const ratingInput = document.getElementById('rating');
    const ratingValue = document.getElementById('rating-value');
    const analyzeBtn = document.getElementById('analyze-btn');
    const reviewText = document.getElementById('review-text');
    const categorySelect = document.getElementById('category');

    const resultSection = document.getElementById('result-section');
    const verdictText = document.getElementById('verdict-text');
    const confidenceBadge = document.getElementById('confidence-badge');
    const probabilityBar = document.getElementById('probability-bar');
    const analysisSummary = document.getElementById('analysis-summary');
    const reasonsList = document.getElementById('reasons-list');

    const loadExampleBtn = document.getElementById('load-example-btn');
    const examplesList = document.getElementById('examples-list');

    // Rating Slider Update
    ratingInput.addEventListener('input', (e) => {
        ratingValue.textContent = parseFloat(e.target.value).toFixed(1);
    });

    // Load Examples Dropdown
    loadExampleBtn.addEventListener('click', () => {
        examplesList.classList.toggle('show');
        if (examplesList.children.length === 0) {
            fetchExamples();
        }
    });

    window.onclick = (e) => {
        if (!e.target.matches('#load-example-btn')) {
            if (examplesList.classList.contains('show')) {
                examplesList.classList.remove('show');
            }
        }
    };

    async function fetchExamples() {
        try {
            const response = await fetch('/demo-cases');
            const data = await response.json();

            examplesList.innerHTML = '';
            data.forEach(example => {
                const div = document.createElement('div');
                div.textContent = example.description;
                div.addEventListener('click', () => {
                    // Populate Form
                    // Remove outer quotes from text if present, as the input field should just have the text
                    // The demo cases have text like "'Actual text'". 
                    let text = example.text;
                    if ((text.startsWith("'") && text.endsWith("'")) || (text.startsWith('"') && text.endsWith('"'))) {
                        text = text.slice(1, -1);
                    }

                    reviewText.value = text;
                    ratingInput.value = example.rating;
                    ratingValue.textContent = example.rating.toFixed(1);
                    categorySelect.value = example.category;
                });
                examplesList.appendChild(div);
            });
        } catch (error) {
            console.error('Error fetching examples:', error);
            examplesList.innerHTML = '<div>Error loading examples</div>';
        }
    }

    // Analyze Logic
    analyzeBtn.addEventListener('click', async () => {
        const text = reviewText.value.trim();
        const rating = parseFloat(ratingInput.value);
        const category = categorySelect.value;

        if (!text) {
            alert('Please enter review text.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';

        // Hide previous results
        resultSection.classList.add('hidden');
        resultSection.classList.remove('is-fake', 'is-genuine');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    rating: rating,
                    category: category
                })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Analysis failed');
            }

            displayResult(result);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Review';
        }
    });

    function displayResult(result) {
        // Prepare UI data
        const isFake = result.label === 'FAKE';
        const probability = result.fake_probability; // 0 to 1
        const percentage = Math.round(probability * 100);

        // Update Sections
        resultSection.classList.remove('hidden');
        resultSection.classList.add(isFake ? 'is-fake' : 'is-genuine');

        verdictText.textContent = result.label;
        confidenceBadge.textContent = `${Math.round(isFake ? percentage : 100 - percentage)}% Confidence`;

        // Animate Bar
        setTimeout(() => {
            probabilityBar.style.width = `${percentage}%`;
        }, 100); // Slight delay for transition

        analysisSummary.textContent = isFake
            ? `Our model detected patterns strongly associated with generic or AI-generated reviews.`
            : `This review exhibits specific details and patterns typical of authentic user experiences.`;

        // Reasons
        reasonsList.innerHTML = '';
        if (result.reasons && Array.isArray(result.reasons)) {
            result.reasons.forEach(reasonStr => {
                // reasonStr format: "'term' (TYPE, score: 0.45)"
                // Parse it roughly
                const match = reasonStr.match(/'(.*)' \((.*), score: (.*)\)/);
                if (match) {
                    const term = match[1];
                    const type = match[2];
                    const score = match[3];

                    const li = document.createElement('li');
                    li.className = type === 'FAKE' ? 'fake-reason' : 'genuine-reason';

                    li.innerHTML = `
                        <span class="reason-term">"${term}"</span>
                        <span class="reason-score">${type} indicator (Impact: ${score})</span>
                    `;
                    reasonsList.appendChild(li);
                } else {
                    // Fallback
                    const li = document.createElement('li');
                    li.textContent = reasonStr;
                    reasonsList.appendChild(li);
                }
            });
        }
    }
});
