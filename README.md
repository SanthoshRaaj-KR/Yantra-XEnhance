# XEnhance: Advanced Image Enhancement with Machine Learning

**XEnhance** is a sophisticated web-based AI solution that leverages deep learning for professional-grade image enhancement.  
Built on a custom **CNN U-Net architecture**, it transforms low-quality or noisy images into high-resolution outputs with remarkable clarity.

---

## 🌟 Features

- 🎯 **Deep Learning Excellence** – Custom-trained CNN U-Net model optimized for image restoration.
- 💻 **Intuitive Web Interface** – Clean, responsive React-based frontend for seamless interaction.
- ⚡ **Real-time Processing** – Experience near-instant enhancement after upload.
- 🔍 **Interactive Comparison** – Side-by-side view of original vs enhanced images.
- 🧃 **Batch Processing** – Upload and enhance multiple images in one go (Pro version).
- 📝 **Suggestions System** – Integrated feedback form with backend storage for user suggestions.

---

## 🚀 Technology Stack

### Frontend

- `React.js` – Component-based dynamic UI  
- `Axios` – Smooth API calls  
- `Tailwind CSS` – Utility-first styling  
- `React Router` – Page navigation

### Backend

- `FastAPI` – Blazing fast Python web framework  
- `TensorFlow / Keras` – Deep Learning backbone  
- `NumPy` – Numerical computation  
- `Pillow` – Image handling  
- `OpenCV` – Vision-based transformations
- `SQLite` – Lightweight database for suggestions storage

---

## 🏗️ Architecture

```
XEnhance/
├── client/                 # React frontend
│   ├── public/             # Static assets
│   └── src/
│       ├── components/     # React components
│       │   └── Suggestions.jsx  # User feedback component
│       ├── hooks/          # Custom React hooks
│       ├── pages/          # Application pages
│       ├── services/       # API integration
│       └── utils/          # Helper functions
├── server/                 # FastAPI backend
│   ├── model/              # U-Net model implementation
│   │   ├── architecture.py # CNN U-Net definition
│   │   └── weights/        # Trained model weights
│   ├── routes/             # API endpoints
│   ├── services/           # Business logic
│   ├── utils/              # Helper functions
│   └── main.py             # FastAPI application entry point
├── suggestions_api/        # Dedicated API for suggestions
│   ├── main.py             # Suggestions API endpoints
│   ├── requirements.txt    # Dependencies
│   └── suggestions.db      # SQLite database (created on first run)
├── uploads/                # Temporary image storage
├── outputs/                # Processed results
├── tests/                  # Test suite
│   ├── frontend/           # Frontend tests
│   └── backend/            # Backend tests
├── docker-compose.yml      # Container orchestration
├── .github/                # GitHub workflows
└── README.md
```

---

## 🧪 How It Works

### Image Enhancement

> Upload → Enhance → Compare

1. 🖼️ **Upload**: Drop your image using the React UI  
2. 🔄 **Preprocessing**: Image normalized for neural net  
3. 🧠 **Model Inference**: CNN U-Net enhances it  
4. 🪄 **Post-processing**: Fine-tuning for optimal clarity  
5. 📤 **Result**: Delivered with original side-by-side

### Suggestions System

> Fill Form → Submit → Store

1. 📝 **User Input**: Collect user feedback through intuitive form
2. ✅ **Validation**: Client-side and server-side validation
3. 💾 **Storage**: Data securely stored in SQLite database
4. 📊 **Admin Access**: Backend API to retrieve all suggestions

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### 🔧 Backend (Image Enhancement)

```bash
git clone https://github.com/yourusername/xenhance.git
cd xenhance/server

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

### 🔧 Backend (Suggestions API)

```bash
cd xenhance/suggestions_api

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### 🖥️ Frontend

```bash
cd xenhance/client
npm install
npm start
```

---

## 🧠 CNN Model Details

The image enhancement feature utilizes a CNN U-Net architecture specifically designed for image restoration tasks. This deep learning model has been trained on thousands of image pairs to learn optimal enhancement patterns.

### 🔧 Model Configuration

The CNN U-Net architecture can be fine-tuned via `server/config.json`:

```json
{
  "model": {
    "input_shape": [256, 256, 3],
    "filters_base": 64,
    "depth": 4,
    "batch_norm": true
  },
  "training": {
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.001
  }
}
```

---

## 📝 Suggestions API

The suggestions system allows users to provide feedback, ask questions, or make feature requests. The API stores this information in a SQLite database.

### API Endpoints

- **POST /api/suggestions/** - Submit a new suggestion
- **GET /api/suggestions/** - Retrieve all suggestions (admin access)

### Data Model

The suggestion form collects:
- First Name
- Last Name
- Email
- Phone Number
- Suggestion text

All data is validated both on the client and server side before storage.

---

## 🚀 Development Notes

### CORS Configuration

For development, the CORS settings allow requests from any origin. For production deployment, update the CORS configuration in `suggestions_api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Security Considerations

For production deployment:
- Add authentication for admin endpoints
- Use environment variables for configuration
- Implement rate limiting
- Configure HTTPS
- Consider data privacy regulations compliance