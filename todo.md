# Mentaid Frontend & Backend Implementation Plan

## Current Phase: Landing Page & Authentication

### Frontend Implementation Plan

#### 1. Frontend Project Structure (Vite + React)
- [ ] Update Frontend Structure
  - [X] Move from viteapp to proper Frontend folder structure
  - [ ] Create proper component organization
  - [ ] Set up Vite configuration for production

#### 2. Landing Page Structure
- [ ] Create React-based landing page with:
  - [ ] Hero section with app description
  - [ ] Login/Signup buttons
  - [ ] Navigation bar
  - [ ] Footer with contact info
  - [ ] Responsive design

#### 3. Authentication Pages
- [ ] Login Page
  - [ ] Username/Email input
  - [ ] Password input
  - [ ] User type selection (Patient/Clinician)
  - [ ] Remember me checkbox
  - [ ] Forgot password link
  - [ ] Login button
  - [ ] Link to signup

- [ ] Signup Page
  - [ ] Username input
  - [ ] Email input
  - [ ] Password input
  - [ ] Confirm password
  - [ ] User type selection (Patient/Clinician)
  - [ ] Terms & Conditions checkbox
  - [ ] Signup button
  - [ ] Link to login

#### 4. Demo Dashboard Pages
- [ ] Patient Dashboard
  - [ ] Mood tracking slider
  - [ ] Journal editor
  - [ ] Profile section
  - [ ] Quick stats

- [ ] Clinician Dashboard
  - [ ] Patient list
  - [ ] Analysis dashboard
  - [ ] Summary view
  - [ ] Search functionality

#### 5. Frontend Technical Setup
- [ ] Project structure
  ```
  Frontend/
  ├── src/
  │   ├── components/
  │   │   ├── auth/
  │   │   ├── layout/
  │   │   ├── common/
  │   │   └── patient/
  │   │   └── clinician/
  │   ├── pages/
  │   │   ├── auth/
  │   │   ├── patient/
  │   │   └── clinician/
  │   ├── services/
  │   │   ├── api/
  │   │   └── auth/
  │   ├── utils/
  │   │   ├── auth.js
  │   │   └── constants.js
  │   └── styles/
  ├── public/
  ├── config/
  │   └── env.d.ts
  └── vite.config.ts
  ```

- [ ] Dependencies (Update package.json)
  - [ ] React
  - [ ] React Router
  - [ ] Axios
  - [ ] Material-UI
  - [ ] React Icons
  - [ ] Formik/Yup
  - [ ] JWT Decode
  - [ ] React Query
  - [ ] Vite
  - [ ] @vercel/analytics

#### 6. Vercel Deployment Configuration
- [ ] Create vercel.json
  ```json
  {
    "version": 2,
    "builds": [
      {
        "src": "package.json",
        "use": "@vercel/static-build"
      }
    ],
    "routes": [
      {
        "src": "/(.*)",
        "dest": "/index.html"
      }
    ],
    "env": {
      "VITE_API_URL": "@VITE_API_URL",
      "VITE_HF_API_KEY": "@VITE_HF_API_KEY"
    }
  }
  ```

### Backend Implementation Plan

#### 1. Authentication API
- [ ] User Model
  ```python
  class User(BaseModel):
      username: str
      email: EmailStr
      password_hash: str
      user_type: str  # "patient" or "clinician"
      created_at: datetime
      updated_at: datetime
  ```

- [ ] Authentication Routes
  - [ ] POST /api/auth/register
  - [ ] POST /api/auth/login
  - [ ] POST /api/auth/refresh-token
  - [ ] POST /api/auth/forgot-password
  - [ ] POST /api/auth/reset-password

- [ ] JWT Authentication
  - [ ] Token generation
  - [ ] Token validation
  - [ ] Refresh token system
  - [ ] Token expiration handling

#### 2. MongoDB Setup
- [ ] Database structure
  ```python
  users_collection = {
      "username": "",
      "email": "",
      "password_hash": "",  # encrypted
      "user_type": "",
      "created_at": "",
      "updated_at": "",
      "last_login": ""
  }
  ```

- [ ] Security measures
  - [ ] Password hashing (bcrypt)
  - [ ] Fernet encryption for sensitive data
  - [ ] Rate limiting
  - [ ] Input validation
  - [ ] CORS configuration

#### 3. Backend Technical Setup
- [ ] Project structure
  ```
  Backend/
  ├── app/
  │   ├── api/
  │   │   ├── routes/
  │   │   │   ├── auth.py
  │   │   │   └── user.py
  │   │   └── dependencies.py
  │   ├── core/
  │   │   ├── config.py
  │   │   ├── security.py
  │   │   └── database.py
  │   ├── models/
  │   │   ├── user.py
  │   │   └── schemas.py
  │   └── utils/
  │       ├── encryption.py
  │       └── auth_utils.py
  └── tests/
  ```

- [ ] Dependencies
  - FastAPI
  - Pydantic
  - Motor (MongoDB async driver)
  - bcrypt
  - python-jose
  - passlib
  - cryptography
  - python-dotenv

### ML Model Deployment Plan (Hugging Face Spaces)

#### 1. Model Setup
- [ ] Create separate ML service
  - [ ] SVM model deployment
  - [ ] MentalBERT deployment
  - [ ] SHAP/LIME implementation
  - [ ] SOAP summary generation

- [ ] Hugging Face Spaces configuration
  - [ ] Set up HF API endpoints
  - [ ] Model versioning
  - [ ] Environment variables
  - [ ] Rate limiting

### Deployment Strategy

#### Frontend (Vercel)
- [ ] Set up Vercel project
- [ ] Configure environment variables
- [ ] Set up build and deploy hooks
- [ ] Configure analytics
- [ ] Set up custom domain

#### Backend (Render)
- [ ] Set up Render deployment
- [ ] Configure environment variables
- [ ] Set up MongoDB connection
- [ ] Configure SSL certificates
- [ ] Set up monitoring

#### ML Models (Hugging Face Spaces)
- [ ] Set up HF Spaces
- [ ] Deploy models
- [ ] Set up API endpoints
- [ ] Configure rate limiting
- [ ] Set up monitoring

### Integration Points

#### Frontend-Backend Integration
- [ ] API endpoints
  - [ ] Authentication endpoints
  - [ ] User management endpoints
  - [ ] Error handling
  - [ ] Loading states
  - [ ] Toast notifications

#### Security Considerations
- [ ] Input validation
- [ ] CSRF protection
- [ ] XSS protection
- [ ] Rate limiting
- [ ] Password complexity
- [ ] Session management

### Next Steps After Authentication
1. Implement Patient Dashboard
2. Implement Clinician Dashboard
3. Add journaling functionality
4. Add mood tracking
5. Add analysis visualization

## Notes
- All components should be reusable and modular
- Follow clean code principles
- Implement proper error handling
- Add comprehensive testing
- Document all API endpoints
