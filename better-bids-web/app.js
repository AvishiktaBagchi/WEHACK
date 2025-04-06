const express = require('express');
const mongoose = require('mongoose');
const path = require('path');

const app = express();
const PORT = 3000;

// MongoDB connection string (replace with yours)
const MONGO_URI = 'mongodb+srv://avibagchi04:QsJdKKFpmmb2oofZ@wehackcluster.hidg3jn.mongodb.net/risk_assessment_db?retryWrites=true&w=majority';


// Connect to MongoDB
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('✅ Connected to MongoDB Atlas'))
.catch(err => console.error('❌ MongoDB connection error:', err));

// Define a simple schema (no strict fields)
const riskSchema = new mongoose.Schema({}, { strict: false });
const Risk = mongoose.model('Risk', riskSchema, 'risk_data');

// Set EJS as the templating engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files like CSS (if any)
app.use(express.static(path.join(__dirname, 'public')));

// Route to render data
app.get('/', async (req, res) => {
  try {
    const riskData = await Risk.find({});
    res.render('index', { riskData });
  } catch (err) {
    console.error(err);
    res.status(500).send('Error fetching data');
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
