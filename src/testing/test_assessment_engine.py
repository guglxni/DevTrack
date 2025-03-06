import unittest
import asyncio
from enhanced_assessment_engine import EnhancedAssessmentEngine, Score

class TestEnhancedAssessmentEngine(unittest.TestCase):
    def setUp(self):
        # Initialize the engine with keyword-based scoring for testing
        self.engine = EnhancedAssessmentEngine(use_embeddings=False)
        self.engine.set_child_age(24)  # Set to 24 months for testing
        
    def test_initialization(self):
        """Test that the engine initializes correctly with milestones"""
        self.assertIsNotNone(self.engine.milestones)
        self.assertGreater(len(self.engine.milestones), 0)
        self.assertEqual(self.engine.child_age, 24)
        
    def test_milestone_filtering(self):
        """Test that milestones are filtered correctly by age"""
        for milestone in self.engine.active_milestones:
            age_range = milestone.age_range
            start_age = int(age_range.split('-')[0].strip())
            end_age = int(age_range.split('-')[1].split()[0].strip())
            
            # All milestones should be appropriate for the child's age (24 months)
            self.assertTrue(
                end_age <= 24 or (start_age <= 24 <= end_age),
                f"Milestone {milestone.behavior} with age range {age_range} not appropriate for age 24 months"
            )
    
    def test_keyword_scoring(self):
        """Test keyword-based scoring"""
        # Get a sample milestone
        milestone = self.engine.active_milestones[0]
        
        # Test various response patterns
        test_cases = [
            ("He cannot do this at all.", Score.CANNOT_DO),
            ("She used to do this but has regressed.", Score.LOST_SKILL),
            ("He's starting to show this sometimes.", Score.EMERGING),
            ("She does this with assistance.", Score.WITH_SUPPORT),
            ("He can do this independently all the time.", Score.INDEPENDENT),
            ("I'm not sure about this one.", Score.NOT_RATED)
        ]
        
        for response, expected_score in test_cases:
            score = self.engine.analyze_response_keywords(response, milestone)
            self.assertEqual(
                score, expected_score,
                f"Expected {expected_score.name} for response '{response}', got {score.name}"
            )
    
    def test_get_next_milestone(self):
        """Test that get_next_milestone returns milestones in the correct order"""
        first_milestone = self.engine.get_next_milestone()
        self.assertIsNotNone(first_milestone)
        
        # The behavior should be marked as asked
        self.assertIn(first_milestone.behavior, self.engine.asked_milestones)
        
        # Get the next milestone
        second_milestone = self.engine.get_next_milestone()
        self.assertIsNotNone(second_milestone)
        
        # The milestones should be different
        self.assertNotEqual(first_milestone.behavior, second_milestone.behavior)
        
        # Both should be marked as asked
        self.assertIn(first_milestone.behavior, self.engine.asked_milestones)
        self.assertIn(second_milestone.behavior, self.engine.asked_milestones)
    
    def test_set_milestone_score(self):
        """Test setting milestone scores"""
        milestone = self.engine.active_milestones[0]
        self.engine.set_milestone_score(milestone, Score.INDEPENDENT)
        
        # The milestone should be marked as assessed
        self.assertIn(milestone.behavior, self.engine.assessed_milestones)
        
        # The score should be stored
        self.assertEqual(self.engine.scores.get(milestone.behavior), Score.INDEPENDENT)
    
    def test_generate_report(self):
        """Test report generation"""
        # Set scores for a few milestones
        for i, milestone in enumerate(self.engine.active_milestones[:5]):
            # Alternate scores for testing
            score = Score.INDEPENDENT if i % 2 == 0 else Score.WITH_SUPPORT
            self.engine.set_milestone_score(milestone, score)
        
        # Generate report
        df, domain_quotients = self.engine.generate_report()
        
        # Verify DataFrame
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Verify domain quotients
        self.assertIsNotNone(domain_quotients)
        self.assertGreater(len(domain_quotients), 0)
        
        # All quotients should be between 0 and 100
        for domain, quotient in domain_quotients.items():
            self.assertGreaterEqual(quotient, 0)
            self.assertLessEqual(quotient, 100)


class TestEnhancedAssessmentEngineAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Initialize the engine with embedding-based scoring for testing
        # Set use_embeddings to False to avoid loading the model during testing
        self.engine = EnhancedAssessmentEngine(use_embeddings=False)
        self.engine.set_child_age(24)  # Set to 24 months for testing
    
    async def test_analyze_response(self):
        """Test analyze_response method"""
        # Get a sample milestone
        milestone = self.engine.active_milestones[0]
        
        # Test response
        response = "She does this with help from me. When I assist her, she can complete the task."
        score = await self.engine.analyze_response(response, milestone)
        
        self.assertEqual(score, Score.WITH_SUPPORT)
    
    async def test_batch_analyze_responses(self):
        """Test batch_analyze_responses method"""
        # Get some sample milestones
        milestones = self.engine.active_milestones[:3]
        
        # Create test responses
        responses = [
            ("He cannot do this yet.", milestones[0]),
            ("She does this with my help.", milestones[1]),
            ("He does this independently all the time.", milestones[2])
        ]
        
        # Expected scores
        expected_scores = [Score.CANNOT_DO, Score.WITH_SUPPORT, Score.INDEPENDENT]
        
        # Test batch scoring
        scores = await self.engine.batch_analyze_responses(responses)
        
        # Verify scores
        self.assertEqual(len(scores), len(expected_scores))
        for i, score in enumerate(scores):
            self.assertEqual(score, expected_scores[i])
    
    async def test_zero_shot_classify(self):
        """Test zero_shot_classify method (falls back to keywords when embeddings disabled)"""
        # Get a sample milestone
        milestone = self.engine.active_milestones[0]
        
        # Test response
        response = "She has mastered this skill completely and does it without any help."
        score = self.engine.zero_shot_classify(response, milestone)
        
        self.assertEqual(score, Score.INDEPENDENT)


if __name__ == '__main__':
    unittest.main() 