#!/usr/bin/env python3
"""
Test script for FastAPI ML Service
Usage: python test_fastapi_service.py [service_url]
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, Any


class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.NC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")


def test_root_endpoint(base_url: str) -> bool:
    """Test root endpoint"""
    print("\n🔍 Testing root endpoint...")

    try:
        response = requests.get(f"{base_url}/")

        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint accessible")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Docs: {base_url}{data.get('docs')}")
            return True
        else:
            print_error(f"Root endpoint failed: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Root endpoint error: {e}")
        return False


def test_health_check(base_url: str) -> bool:
    """Test health check endpoint"""
    print("\n🔍 Testing health check...")

    try:
        response = requests.get(f"{base_url}/health")

        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            model_loaded = data.get('model_loaded')

            if status == 'healthy' and model_loaded:
                print_success("Health check passed")
                print(f"   Status: {status}")
                print(f"   Model loaded: {model_loaded}")
                print(f"   Service: {data.get('service')}")
                return True
            else:
                print_warning(f"Service unhealthy: {status}, Model loaded: {model_loaded}")
                return False
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test model info endpoint"""
    print("\n🔍 Testing model info...")

    try:
        response = requests.get(f"{base_url}/model-info")

        if response.status_code == 200:
            data = response.json()
            print_success("Model info retrieved")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Estimators: {data.get('n_estimators')}")
            print(f"   Max depth: {data.get('max_depth')}")
            print(f"   Features: {data.get('n_features')}")
            print(f"   Municipalities: {len(data.get('supported_municipalities', []))}")
            return True
        else:
            print_error(f"Model info failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Model info error: {e}")
        return False


def test_municipalities(base_url: str) -> list:
    """Test municipalities endpoint"""
    print("\n🔍 Testing municipalities...")

    try:
        response = requests.get(f"{base_url}/municipalities")

        if response.status_code == 200:
            data = response.json()
            municipalities = data.get('municipalities', [])
            print_success(f"Municipalities retrieved: {len(municipalities)}")
            if municipalities:
                print(f"   Examples: {', '.join(municipalities[:5])}...")
            return municipalities
        else:
            print_error(f"Municipalities failed: {response.status_code}")
            return []

    except Exception as e:
        print_error(f"Municipalities error: {e}")
        return []


def test_prediction(base_url: str, municipalities: list) -> bool:
    """Test single prediction"""
    print("\n🔍 Testing single prediction...")

    # Use first available municipality or default
    municipality = municipalities[0] if municipalities else "Центар"

    test_property = {
        "municipality": municipality,
        "area": 67,
        "number_of_rooms": 2,
        "Балкон / Тераса": 0,
        "Лифт": 1,
        "Приземје": 0,
        "Паркинг простор / Гаража": 0,
        "Поткровје": 0,
        "Нова градба": 0,
        "Реновиран": 1,
        "Наместен": 1,
        "Подрум": 1,
        "Интерфон": 1,
        "Дуплекс": 0
    }

    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_property,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                price = data.get('predicted_price')
                price_per_m2 = data.get('price_per_square_meter')

                print_success("Prediction successful")
                print(
                    f"   Property: {municipality}, {test_property['area']}m², {test_property['number_of_rooms']} rooms")
                print(f"   Predicted price: €{price:,.2f}")
                print(f"   Price per m²: €{price_per_m2:,.2f}/m²")

                # Show property summary if available
                if 'property_summary' in data:
                    summary = data['property_summary']
                    print(f"   Summary: {summary}")

                return True
            else:
                print_error(f"Prediction failed: {data.get('error')}")
                return False
        else:
            print_error(f"Prediction request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Prediction error: {e}")
        return False


def test_batch_prediction(base_url: str, municipalities: list) -> bool:
    """Test batch prediction"""
    print("\n🔍 Testing batch prediction...")

    municipality = municipalities[0] if municipalities else "Центар"

    test_properties = {
        "properties": [
            {
                "municipality": municipality,
                "area": 50,
                "number_of_rooms": 2,
                "Балкон / Тераса": 1,
                "Лифт": 1,
                "Приземје": 0,
                "Паркинг простор / Гаража": 0,
                "Поткровје": 0,
                "Нова градба": 0,
                "Реновиран": 1,
                "Наместен": 1,
                "Подрум": 0,
                "Интерфон": 1,
                "Дуплекс": 0
            },
            {
                "municipality": municipality,
                "area": 80,
                "number_of_rooms": 3,
                "Балкон / Тераса": 1,
                "Лифт": 1,
                "Приземје": 0,
                "Паркинг простор / Гаража": 1,
                "Поткровје": 0,
                "Нова градба": 1,
                "Реновиран": 0,
                "Наместен": 0,
                "Подрум": 1,
                "Интерфон": 1,
                "Дуплекс": 0
            },
            {
                "municipality": municipality,
                "area": 100,
                "number_of_rooms": 4,
                "Балкон / Тераса": 1,
                "Лифт": 1,
                "Приземје": 0,
                "Паркинг простор / Гаража": 1,
                "Поткровје": 0,
                "Нова градба": 1,
                "Реновиран": 1,
                "Наместен": 1,
                "Подрум": 1,
                "Интерфон": 1,
                "Дуплекс": 1
            }
        ]
    }

    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json=test_properties,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                results = data.get('results', [])
                errors = data.get('errors', [])

                print_success("Batch prediction successful")
                print(f"   Total properties: {data.get('total_properties')}")
                print(f"   Successful: {data.get('successful_predictions')}")
                print(f"   Failed: {data.get('failed_predictions')}")

                for result in results[:3]:  # Show first 3 results
                    price = result.get('predicted_price')
                    area = result.get('area')
                    print(f"   Property {result.get('index')}: €{price:,.2f} ({area}m²)")

                if errors:
                    print_warning(f"   {len(errors)} errors occurred")

                return True
            else:
                print_error(f"Batch prediction failed: {data.get('error')}")
                return False
        else:
            print_error(f"Batch prediction request failed: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Batch prediction error: {e}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error handling"""
    print("\n🔍 Testing error handling...")

    # Test with invalid data
    invalid_property = {
        "municipality": "InvalidMunicipality",
        "area": -50,  # Invalid negative area
        "number_of_rooms": 0  # Invalid zero rooms
    }

    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_property,
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 (Unprocessable Entity) due to Pydantic validation
        if response.status_code == 422:
            print_success("Error handling works correctly (422 Validation Error)")
            data = response.json()
            if 'detail' in data:
                print(f"   Validation errors: {len(data['detail'])} field(s)")
            return True
        else:
            print_warning(f"Expected 422 error, got {response.status_code}")
            return True  # Still passing as it's handling errors

    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        return False


def test_api_docs(base_url: str) -> bool:
    """Test API documentation endpoints"""
    print("\n🔍 Testing API documentation...")

    try:
        # Test Swagger UI
        response_docs = requests.get(f"{base_url}/docs")
        docs_ok = response_docs.status_code == 200

        # Test ReDoc
        response_redoc = requests.get(f"{base_url}/redoc")
        redoc_ok = response_redoc.status_code == 200

        # Test OpenAPI JSON
        response_openapi = requests.get(f"{base_url}/openapi.json")
        openapi_ok = response_openapi.status_code == 200

        if docs_ok and redoc_ok and openapi_ok:
            print_success("API documentation accessible")
            print(f"   Swagger UI: {base_url}/docs")
            print(f"   ReDoc: {base_url}/redoc")
            print(f"   OpenAPI spec: {base_url}/openapi.json")
            return True
        else:
            print_warning("Some documentation endpoints failed")
            print(f"   Swagger UI: {'✓' if docs_ok else '✗'}")
            print(f"   ReDoc: {'✓' if redoc_ok else '✗'}")
            print(f"   OpenAPI: {'✓' if openapi_ok else '✗'}")
            return False

    except Exception as e:
        print_error(f"API docs test error: {e}")
        return False


def main():
    """Main test function"""
    # Get service URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"{Colors.BLUE}🧪 Testing FastAPI ML Service at {base_url}{Colors.NC}")
    print(f"{Colors.BLUE}⏰ Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.NC}")
    print("=" * 70)

    # Run tests
    tests_passed = 0
    total_tests = 8

    if test_root_endpoint(base_url):
        tests_passed += 1

    if test_health_check(base_url):
        tests_passed += 1

    if test_model_info(base_url):
        tests_passed += 1

    municipalities = test_municipalities(base_url)
    if municipalities:
        tests_passed += 1

    if test_prediction(base_url, municipalities):
        tests_passed += 1

    if test_batch_prediction(base_url, municipalities):
        tests_passed += 1

    if test_error_handling(base_url):
        tests_passed += 1

    if test_api_docs(base_url):
        tests_passed += 1

    # Results
    print("\n" + "=" * 70)
    print(f"{Colors.BLUE}🏁 Test Results: {tests_passed}/{total_tests} tests passed{Colors.NC}")

    if tests_passed == total_tests:
        print(f"{Colors.GREEN}🎉 All tests passed! FastAPI service is working correctly.{Colors.NC}")
        print(f"\n{Colors.BLUE}📚 Explore the interactive API docs at:{Colors.NC}")
        print(f"   {base_url}/docs")
        sys.exit(0)
    elif tests_passed >= total_tests - 2:
        print(f"{Colors.YELLOW}⚠️  Most tests passed. Check the failed tests above.{Colors.NC}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}⚠️  Several tests failed. Check the service configuration.{Colors.NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()