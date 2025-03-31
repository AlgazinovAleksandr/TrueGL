// script.js
async function searchTruth() {
    const query = document.getElementById('searchInput').value;
    const contentArea = document.getElementById('contentArea');
    const truthFill = document.getElementById('truthFill');

    try {
        // Fetch search results from the backend
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        // Update UI with results
        contentArea.innerHTML = `
            <h2>Results for "${data.query}"</h2>
            ${data.results.length ? 
                data.results.map(result => `
                    <div class="result">
                        <a href="${result.url}" target="_blank">${result.title}</a>
                        <p>${result.content}</p>
                    </div>
                `).join('') : 
                '<p>No results found</p>'}
        `;

        // Update truth meter
        const truthScore = data.truthScore; // Comes from backend (still random for now)
        truthFill.style.width = `${truthScore}%`;
        
        // Change truth bar color based on score
        if (truthScore > 70) {
            truthFill.style.background = '#4CAF50'; // Green
        } else if (truthScore > 30) {
            truthFill.style.background = '#FFC107'; // Yellow
        } else {
            truthFill.style.background = '#F44336'; // Red
        }
    } catch (error) {
        // Handle errors gracefully
        contentArea.innerHTML = `
            <h2>Results for "${query}"</h2>
            <p>Error loading results. Please try again.</p>
        `;
        truthFill.style.width = '0%';
        console.error('Search failed:', error);
    }
}

// Add tab functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        document.querySelector('.tab.active')?.classList.remove('active');
        this.classList.add('active');
    });
});