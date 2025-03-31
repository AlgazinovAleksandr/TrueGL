const express = require('express');
const { crawlPage, search } = require('./crawler');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

// Search endpoint
app.get('/api/search', (req, res) => {
    const query = req.query.q || '';
    const results = search(query);
    res.json({
        query: query,
        results: results,
        truthScore: Math.floor(Math.random() * 100) // Placeholder
    });
});

// Start crawling on server start (optional)
crawlPage('https://example.com', 5) // Crawl 5 pages from example.com
    .then(() => console.log('Crawling complete'))
    .catch(err => console.error('Crawling failed:', err));

app.listen(port, () => {
    console.log(`Trugle server running at http://localhost:${port}`);
});