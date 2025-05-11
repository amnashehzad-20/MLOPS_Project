import React, { useState } from 'react';
import { FileText, Camera, Pen, MessageSquare, Image } from 'lucide-react';

export default function NewspaperClassifier() {
  const [features, setFeatures] = useState({
    sentiment_score: '',
    keyword_count: '',
    headline_length: '',
    readability_score: '',
    image_count: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    setFeatures({
      ...features,
      [e.target.name]: e.target.value
    });
    setError(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    
    // Validate inputs
    const values = Object.values(features);
    if (values.some(val => val === '' || isNaN(parseFloat(val)))) {
      setError('Please fill in all fields with valid numbers');
      setLoading(false);
      return;
    }

    // Simulate API call to model
    setTimeout(() => {
      // Mock prediction logic based on feature values
      const featureSum = Object.values(features).reduce((sum, val) => sum + parseFloat(val), 0);
      let category;
      
      if (featureSum < 200) {
        category = 0; // Short
      } else if (featureSum < 400) {
        category = 1; // Medium
      } else {
        category = 2; // Long
      }
      
      setPrediction(category);
      setLoading(false);
    }, 1000);
  };

  const getCategoryInfo = (category) => {
    const categories = {
      0: { label: 'Brief', subtitle: 'Quick Read Edition', description: 'Concise reporting for the busy reader', icon: '📰', timing: '2-5 minutes' },
      1: { label: 'Feature', subtitle: 'Standard Edition', description: 'Comprehensive coverage with balanced depth', icon: '📄', timing: '5-10 minutes' },
      2: { label: 'Investigation', subtitle: 'Sunday Edition', description: 'In-depth analysis and investigative journalism', icon: '📚', timing: '10+ minutes' }
    };
    return categories[category] || categories[0];
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
                        <Pen className="w-4 h-4" />
                        Sentiment Analysis
                      </span>
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      name="sentiment_score"
                      value={features.sentiment_score}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Range: -1 to 1"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Editorial tone measurement</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <MessageSquare className="w-4 h-4" />
                        Keyword Density
                      </span>
                    </label>
                    <input
                      type="number"
                      name="keyword_count"
                      value={features.keyword_count}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Total keywords found"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">SEO optimization metric</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <FileText className="w-4 h-4" />
                        Headline Characters
                      </span>
                    </label>
                    <input
                      type="number"
                      name="headline_length"
                      value={features.headline_length}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Character count"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Above the fold impact</span>
                    </label>
                  </div>
                </div>

                {/* Column 2 */}
                <div className="space-y-6">
                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <FileText className="w-4 h-4" />
                        Readability Index
                      </span>
                    </label>
                    <input
                      type="number"
                      name="readability_score"
                      value={features.readability_score}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Flesch score (0-100)"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Reader comprehension level</span>
                    </label>
                  </div>

                  <div className="form-control border-b border-base-300 pb-4 space-y-3 space-x-3">
                    <label className="label">
                      <span className="label-text flex items-center gap-2 text-sm font-bold uppercase tracking-wide">
                        <Image className="w-4 h-4" />
                        Visual Content
                      </span>
                    </label>
                    <input
                      type="number"
                      name="image_count"
                      value={features.image_count}
                      onChange={handleInputChange}
                      className="input input-bordered border-1 rounded-sm w-100 font-serif text-lg"
                      placeholder="Number of images"
                    />
                    <label className="label">
                      <span className="label-text-alt italic">Photography & illustrations</span>
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
                    <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
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
          {prediction !== null && (
            <div className="mt-8">
              <div className="card bg-neutral text-neutral-content shadow-2xl">
                <div className="card-body p-8">
                  <h3 className="text-3xl font-bold text-center mb-6 tracking-wider">
                    🎯 CLASSIFICATION BULLETIN 🎯
                  </h3>
                  
                  <div className="card bg-base-100 text-base-content vintage-frame">
                    <div className="card-body text-center p-8">
                      <div className="text-8xl mb-4">{getCategoryInfo(prediction).icon}</div>
                      <h4 className="text-4xl font-black uppercase tracking-wider mb-2">
                        {getCategoryInfo(prediction).label}
                      </h4>
                      <p className="text-xl font-serif italic mb-4">
                        {getCategoryInfo(prediction).subtitle}
                      </p>
                      <div className="divider"></div>
                      <p className="text-lg font-serif leading-relaxed mb-4">
                        {getCategoryInfo(prediction).description}
                      </p>
                      <p className="text-sm font-serif uppercase tracking-wide text-base-content/70">
                        Estimated Read Time: {getCategoryInfo(prediction).timing}
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 flex justify-center">
                    <div className="card bg-warning text-warning-content">
                      <div className="card-body p-4 text-center">
                        <p className="font-bold">★ EXTRA! EXTRA! ★</p>
                        <p className="font-serif text-sm">Classification confidence: High</p>
                      </div>
                    </div>
                  </div>
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
              <h5 className="font-bold uppercase tracking-wider mb-2">Weather</h5>
              <p className="font-serif text-sm">Cloudy with a chance of algorithms</p>
            </div>
            <div>
              <h5 className="font-bold uppercase tracking-wider mb-2">Stock Market</h5>
              <p className="font-serif text-sm">ML Accuracy ↑ 98.5%</p>
            </div>
            <div>
              <h5 className="font-bold uppercase tracking-wider mb-2">Today's Puzzle</h5>
              <p className="font-serif text-sm">What has features but no face?</p>
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