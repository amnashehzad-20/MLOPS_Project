import React, { useState, useEffect } from 'react';
import { FileText, Clock, User, Calendar, Hash, AlertCircle, RefreshCw } from 'lucide-react';
import API_CONFIG from '../api/config';

export default function NewspaperClassifier() {
  const [features, setFeatures] = useState({
    title_length: '',
    description_length: '',
    has_author: 0,
    source_category: 0,
    publish_hour: '',
    publish_day: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  // Check API health and get model info on mount
  useEffect(() => {
    checkHealth();
    getModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/health`, {
        headers: API_CONFIG.HEADERS
      });
      const data = await response.json();
      setHealthStatus(data);
    } catch (err) {
      console.error('Health check failed:', err);
    }
  };

  const getModelInfo = async () => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/model-info`, {
        headers: API_CONFIG.HEADERS
      });
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('Failed to get model info:', err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFeatures({
      ...features,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    });
    setError(null);
  };

  const validateInputs = () => {
    // Check required fields
    if (!features.title_length || !features.publish_hour || !features.publish_day) {
      return 'Please fill in title length, publish hour, and publish day.';
    }

    // Validate ranges
    const hour = parseInt(features.publish_hour);
    const day = parseInt(features.publish_day);
    
    if (hour < 0 || hour > 23) {
      return 'Publish hour must be between 0 and 23';
    }
    
    if (day < 1 || day > 31) {
      return 'Publish day must be between 1 and 31';
    }
    
    return null;
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    
    // Validate inputs
    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      setLoading(false);
      return;
    }

    try {
      const requestData = {
        title_length: parseInt(features.title_length) || 0,
        description_length: parseInt(features.description_length) || 0,
        has_author: features.has_author,
        source_category: parseInt(features.source_category) || 0,
        publish_hour: parseInt(features.publish_hour),
        publish_day: parseInt(features.publish_day),
        debug: true // Enable debug info for development
      };

      const response = await fetch(`${API_CONFIG.BASE_URL}/predict`, {
        method: 'POST',
        headers: API_CONFIG.HEADERS,
        body: JSON.stringify(requestData),
        timeout: API_CONFIG.TIMEOUT
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setPrediction(data);
      
    } catch (err) {
      setError(err.message || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleReloadModel = async () => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/reload-model`, {
        method: 'POST',
        headers: API_CONFIG.HEADERS
      });
      const data = await response.json();
      if (response.ok) {
        setError(null);
        getModelInfo(); // Refresh model info
      } else {
        setError('Failed to reload model');
      }
    } catch (err) {
      setError('Failed to reload model');
    }
  };

  const getCategoryInfo = (category) => {
    const categories = {
      'Short': { 
        label: 'Brief', 
        subtitle: 'Quick Read Edition', 
        description: 'Concise reporting for the busy reader', 
        icon: '📰', 
        timing: '2-5 minutes' 
      },
      'Medium': { 
        label: 'Feature', 
        subtitle: 'Standard Edition', 
        description: 'Comprehensive coverage with balanced depth', 
        icon: '📄', 
        timing: '5-10 minutes' 
      },
      'Long': { 
        label: 'Investigation', 
        subtitle: 'Sunday Edition', 
        description: 'In-depth analysis and investigative journalism', 
        icon: '📚', 
        timing: '10+ minutes' 
      }
    };
    return categories[category] || categories['Short'];
  };

  const currentDate = new Date().toLocaleDateString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50 via-amber-50 to-yellow-50" data-theme="newspaper">
      {/* Newspaper Header */}
      <header className="bg-gradient-to-b from-zinc-900 to-zinc-800 text-amber-50 py-8 newspaper-header">
        <div className="container mx-auto px-4">
          <div className="text-center mb-4">
            <h1 className="text-6xl font-bold tracking-tighter mb-2 newspaper-title">
              The Daily Classifier
            </h1>
            <div className="flex justify-center items-center gap-8 text-sm tracking-wide">
              <span>Est. 2025</span>
              <span className="text-amber-300">★ ★ ★</span>
              <span>{currentDate}</span>
              <span className="text-amber-300">★ ★ ★</span>
              <span>Vol. ML, No. 42</span>
            </div>
          </div>
          <div className="border-t border-b border-amber-200/30 py-2 mt-4">
            <p className="text-center text-lg font-serif italic text-amber-200">
              "All the News That's Fit to Classify" — Machine Learning Edition
            </p>
          </div>
        </div>
      </header>

      {/* Status Bar */}
      <div className="bg-zinc-700 text-amber-50 py-2">
        <div className="container mx-auto px-4 flex justify-between items-center text-sm">
          <div className="flex items-center gap-4">
            <span className={`badge ${healthStatus?.status === 'Service healthy' ? 'badge-success' : 'badge-warning'}`}>
              {healthStatus?.status || 'Checking...'}
            </span>
            {modelInfo && (
              <span>Model v{modelInfo.version} ({modelInfo.status})</span>
            )}
          </div>
          <button 
            onClick={handleReloadModel}
            className="btn btn-sm btn-ghost flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Reload Model
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-5xl mx-auto">
          {/* Front Page Notice */}
          <div className="card bg-base-100 shadow-xl mb-8 newspaper-border">
            <div className="card-body text-center p-6">
              <h2 className="text-3xl font-bold mb-4">
                ⚡ BREAKING NEWS ⚡
              </h2>
              <p className="text-xl font-serif leading-relaxed text-neutral-content">
                Local Machine Learning Model Successfully Predicts Article Length Categories with Remarkable Accuracy
              </p>
              <div className="divider"></div>
              <p className="text-sm font-serif italic text-neutral-content/80">
                Submit your article metrics below for instant classification
              </p>
            </div>
          </div>

          {/* Input Form styled as newspaper columns */}
          <div className="card bg-base-100 shadow-xl newspaper-border">
            <div className="card-body p-8">
              <h3 className="text-2xl font-bold text-center uppercase tracking-wider mb-6 pb-2 border-b-2 border-neutral">
                Article Metrics Department
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Column 1 */}
                <div className="space-y-6">
                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <FileText className="w-4 h-4" />
                        Title Length *
                      </span>
                    </label>
                    <input
                      type="number"
                      name="title_length"
                      value={features.title_length}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Number of characters (17-210)"
                      required
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Length of the article title in characters</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <FileText className="w-4 h-4" />
                        Description Length
                      </span>
                    </label>
                    <input
                      type="number"
                      name="description_length"
                      value={features.description_length}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Number of characters (0-260)"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Length of the article description</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label flex items-center cursor-pointer">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <User className="w-4 h-4" />
                        Has Author? *
                      </span>
                      <input
                        type="checkbox"
                        name="has_author"
                        checked={features.has_author === 1}
                        onChange={handleInputChange}
                        className="checkbox checkbox-accent"
                      />
                    </label>
                    <label className="label">
                      <span className="label-text-alt italic">Whether the article has an attributed author</span>
                    </label>
                  </div>
                </div>

                {/* Column 2 */}
                <div className="space-y-6">
                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <Hash className="w-4 h-4" />
                        Source Category
                      </span>
                    </label>
                    <select
                      name="source_category"
                      value={features.source_category}
                      onChange={handleInputChange}
                      className="select select-bordered border-1 rounded-sm w-100 font-serif text-lg"
                    >
                      <option value={0}>Category 0</option>
                      <option value={1}>Category 1</option>
                      <option value={2}>Category 2</option>
                      <option value={3}>Category 3</option>
                    </select>
                    <label className="label">
                      <span className="label-text-alt italic">Article category classification</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <Clock className="w-4 h-4" />
                        Publish Hour *
                      </span>
                    </label>
                    <input
                      type="number"
                      name="publish_hour"
                      value={features.publish_hour}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="0-23"
                      min="0"
                      max="23"
                      required
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Hour of publication (24-hour format)</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <Calendar className="w-4 h-4" />
                        Publish Day *
                      </span>
                    </label>
                    <input
                      type="number"
                      name="publish_day"
                      value={features.publish_day}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="1-31"
                      min="1"
                      max="31"
                      required
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Day of the month (1-31)</span>
                    </label>
                  </div>

                  {/* Submit Button */}
                  <div className="pt-4">
                    <button 
                      onClick={handleSubmit}
                      className={`btn btn-neutral btn-block uppercase tracking-widest font-bold ${loading ? 'loading' : ''}`}
                      disabled={loading}
                    >
                      {loading ? 'Processing...' : 'Classify Article'}
                    </button>
                  </div>
                </div>
              </div>

              {error && (
                <div className="alert alert-error newspaper-border">
                  <div>
                    <AlertCircle className="h-6 w-6" />
                    <div className="text-center w-full">
                      <p className="font-bold">⚠ CORRECTION NEEDED ⚠</p>
                      <p className="font-serif">{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Prediction Result styled as newspaper announcement */}
          {prediction && (
            <div className="mt-8">
              <div className="card bg-neutral text-neutral-content shadow-2xl">
                <div className="card-body p-8">
                  <h3 className="text-3xl font-bold text-center mb-6 tracking-wider">
                    🎯 CLASSIFICATION BULLETIN 🎯
                  </h3>
                  
                  <div className="card bg-base-100 text-base-content vintage-frame">
                    <div className="card-body text-center p-8">
                      <div className="text-8xl mb-4">{getCategoryInfo(prediction.prediction).icon}</div>
                      <h4 className="text-4xl font-black uppercase tracking-wider mb-2">
                        {getCategoryInfo(prediction.prediction).label}
                      </h4>
                      <p className="text-xl font-serif italic mb-4">
                        {getCategoryInfo(prediction.prediction).subtitle}
                      </p>
                      <div className="divider"></div>
                      <p className="text-lg font-serif leading-relaxed mb-4">
                        {getCategoryInfo(prediction.prediction).description}
                      </p>
                      <p className="text-sm font-serif uppercase tracking-wide text-base-content/70">
                        Estimated Read Time: {getCategoryInfo(prediction.prediction).timing}
                      </p>
                    </div>
                  </div>

                  {/* Warning about data imbalance */}
                  {prediction.warning && (
                    <div className="mt-6 flex justify-center">
                      <div className="card bg-warning text-warning-content">
                        <div className="card-body p-4 text-center">
                          <p className="font-bold">★ EDITORIAL NOTE ★</p>
                          <p className="font-serif text-sm">{prediction.warning}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Probability display */}
                  {prediction.probabilities && (
                    <div className="mt-6 flex justify-center">
                      <div className="card bg-base-300">
                        <div className="card-body p-4">
                          <h5 className="font-bold text-center mb-2">Classification Confidence</h5>
                          <div className="space-y-2">
                            {Object.entries(prediction.probabilities).map(([category, probability]) => (
                              <div key={category} className="flex justify-between items-center">
                                <span className="font-serif">{category}:</span>
                                <span className="font-mono">{(probability * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Debug info */}
                  {prediction.debug && (
                    <details className="mt-6">
                      <summary className="cursor-pointer font-bold text-sm uppercase">
                        Technical Details (Debug Mode)
                      </summary>
                      <div className="mt-2 p-4 bg-base-300 rounded-lg">
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(prediction.debug, null, 2)}
                        </pre>
                      </div>
                    </details>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer styled as newspaper footer */}
      <footer className="mt-16 bg-neutral text-neutral-content py-8 newspaper-footer">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div>
              <h5 className="font-bold uppercase tracking-wider mb-2">API Status</h5>
              <p className="font-serif text-sm">{healthStatus?.status || 'Checking...'}</p>
            </div>
            <div>
              <h5 className="font-bold uppercase tracking-wider mb-2">Model Version</h5>
              <p className="font-serif text-sm">{modelInfo ? `v${modelInfo.version}` : 'Loading...'}</p>
            </div>
            <div>
              <h5 className="font-bold uppercase tracking-wider mb-2">Prediction Engine</h5>
              <p className="font-serif text-sm">MLflow + scikit-learn</p>
            </div>
          </div>
          <div className="divider divider-neutral"></div>
          <p className="text-center font-serif text-sm">
            © 2025 The Daily Classifier | Published by Machine Learning Press | All Rights Reserved
          </p>
        </div>
      </footer>
    </div>
  );
}