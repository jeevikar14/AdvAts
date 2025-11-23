#!/usr/bin/env python3
"""
System test script - validates all components
"""
import sys

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    try:
        from inference import RobustATSInference
        from gemini_integration import get_gemini_client
        from job_recommender import EnhancedJobRecommender
        from utils import FileParser, DatabaseManager
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ats_inference():
    """Test ATS inference system"""
    print("\n🧪 Testing ATS inference...")
    try:
        from inference import RobustATSInference
        
        ats = RobustATSInference()
        
        # Test feature extraction
        resume = "Python developer with 5 years experience in machine learning and AWS"
        jd = "Looking for senior Python developer with ML experience"
        
        features, resume_emb, jd_emb = ats.extract_features(resume, jd)
        
        if features is None:
            print("❌ Feature extraction failed")
            return False
        
        # Test prediction
        score = ats.predict_ats_score(features)
        
        if score < 0 or score > 100:
            print(f"❌ Invalid score: {score}")
            return False
        
        print(f"✅ ATS Score: {score:.1f}/100")
        print(f"✅ Category: {ats.get_ats_category(score)}")
        print(f"✅ Model Status: {ats.get_model_status()}")
        
        return True
    except Exception as e:
        print(f"❌ ATS test error: {e}")
        return False

def test_gemini():
    """Test Gemini integration"""
    print("\n🧪 Testing Gemini integration...")
    try:
        from gemini_integration import get_gemini_client
        
        gemini = get_gemini_client()
        
        # Test feedback generation
        feedback = gemini.generate_resume_feedback(
            "Sample resume", "Sample JD", 75, [75, 70, 65, 80, 5]
        )
        
        if not feedback or len(feedback) < 10:
            print("❌ Feedback generation failed")
            return False
        
        print(f"✅ Feedback generated ({len(feedback)} chars)")
        
        # Test chat
        response = gemini.chat_assistant("Tell me about resume tips", [])
        
        if not response or len(response) < 10:
            print("❌ Chat failed")
            return False
        
        print(f"✅ Chat working ({len(response)} chars)")
        
        return True
    except Exception as e:
        print(f"❌ Gemini test error: {e}")
        return False

def test_job_recommender():
    """Test job recommender"""
    print("\n🧪 Testing job recommender...")
    try:
        from job_recommender import EnhancedJobRecommender
        
        recommender = EnhancedJobRecommender()
        
        # Test recommendations
        jobs = recommender.get_job_recommendations(
            ['Python', 'Machine Learning'], 3, 'Remote', 3
        )
        
        if not jobs or len(jobs) == 0:
            print("❌ No jobs returned")
            return False
        
        print(f"✅ Found {len(jobs)} job recommendations")
        print(f"✅ Sample: {jobs[0]['title']} at {jobs[0]['company']}")
        
        return True
    except Exception as e:
        print(f"❌ Job recommender test error: {e}")
        return False

def test_database():
    """Test database setup"""
    print("\n🧪 Testing database...")
    try:
        from setup_database import setup_database
        
        setup_database()
        print("✅ Database initialized")
        
        return True
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 ATS SYSTEM VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("ATS Inference", test_ats_inference),
        ("Gemini Integration", test_gemini),
        ("Job Recommender", test_job_recommender),
        ("Database", test_database)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("=" * 60)
    print(f"Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR SUBMISSION!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed - Review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())