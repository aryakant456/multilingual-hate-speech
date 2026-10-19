# Frontend (React + Vite)

```
cd frontend
npm install        # once
npm run dev        # http://localhost:5173   (needs the backend on port 8000)
```
The browser calls `/api/predict`; Vite forwards it to the FastAPI backend (`vite.config.js`).
