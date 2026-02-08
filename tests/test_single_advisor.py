#!/usr/bin/env python3
"""Test single advisor with improved JSON parsing."""

import asyncio
import json
import pytest
import sys
from enum import Enum

sys.path.append('.')

from app.config import settings

class AdvisorType(str, Enum):
    VALUE = "value"
    MACRO = "macro" 

@pytest.mark.asyncio
async def test_advisor_with_parsing(advisor_type: AdvisorType):
    """Test a single advisor with improved JSON parsing."""
    print(f"🧪 Testing {advisor_type.value.upper()} Advisor")
    print("=" * 50)
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        
        # Map advisor types to configuration keys
        advisor_to_config_key = {
            AdvisorType.VALUE: "VALUE_INVESTOR",
            AdvisorType.MACRO: "MACRO_STRATEGIST"
        }
        
        config_key = advisor_to_config_key[advisor_type]
        model = settings.advisor_models[config_key]
        
        prompt = """You are a specialist equity researcher. Your job is to propose exactly **ten** high-upside stocks for an aggressive growth portfolio.

**Output valid JSON only**—no markdown, no prose, **no enclosing object**.
**Your entire response must be exactly a JSON array of ten objects**, starting with `[` and ending with `]`.

Example format:
[
  {
    "ticker": "TSLA",
    "company": "Tesla Inc.",
    "allocation": 12,
    "justification": "Short reason",
    "confidence": 0.82,
    "risk": 0.74,
    "sell_trigger": "If delayed > 2 yrs"
  }
]

Allocations must sum to 100. Keep justification ≤ 40 words. Provide exactly 10 stocks."""

        print(f"🔄 Calling {model}...")
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7,
            timeout=60.0
        )
        
        content = response.choices[0].message.content.strip()
        print(f"✅ Received response")
        print(f"📝 Raw response (first 200 chars): {content[:200]}...")
        
        # Parse JSON response with markdown handling
        try:
            # Clean up markdown formatting if present
            clean_content = content.strip()
            
            # Remove markdown code blocks if present
            if clean_content.startswith('```json'):
                clean_content = clean_content[7:]  # Remove ```json
                print("🧹 Removed ```json prefix")
            elif clean_content.startswith('```'):
                clean_content = clean_content[3:]   # Remove ```
                print("🧹 Removed ``` prefix")
            
            if clean_content.endswith('```'):
                clean_content = clean_content[:-3]  # Remove trailing ```
                print("🧹 Removed ``` suffix")
            
            clean_content = clean_content.strip()
            print(f"🧹 Cleaned content (first 200 chars): {clean_content[:200]}...")
            
            recommendations = json.loads(clean_content)
            
            if isinstance(recommendations, list) and len(recommendations) == 10:
                print(f"✅ Successfully parsed {len(recommendations)} recommendations!")
                total_allocation = sum(rec.get('allocation', 0) for rec in recommendations)
                print(f"📊 Total allocation: {total_allocation}%")
                
                print("\nTop 3 recommendations:")
                for rec in recommendations[:3]:
                    print(f"  • {rec.get('ticker', 'N/A')}: {rec.get('allocation', 0)}% - {rec.get('justification', 'N/A')[:50]}...")
                
                return True
            else:
                print(f"❌ Invalid format: expected 10 items, got {len(recommendations) if isinstance(recommendations, list) else 'non-list'}")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing still failed: {e}")
            print(f"Final cleaned content: {clean_content[:500]}...")
            return False
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

async def main():
    """Test the problematic advisors."""
    print("🔧 Testing Advisors with JSON Parsing Fix")
    print("=" * 60)
    
    advisors_to_test = [AdvisorType.VALUE, AdvisorType.MACRO]
    
    success_count = 0
    for advisor in advisors_to_test:
        if await test_advisor_with_parsing(advisor):
            success_count += 1
        print()  # Add spacing
    
    print(f"📊 Results: {success_count}/{len(advisors_to_test)} advisors working")

if __name__ == "__main__":
    asyncio.run(main()) 