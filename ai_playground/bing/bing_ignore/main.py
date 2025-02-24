import ai_playground.bing.bing_before.data_fetcher as data_fetcher  # 1번 파일 실행 (환경변수 설정됨)
import curation_generator  # 2번 파일 실행 (환경변수 활용 가능)

# 실행 흐름 정의
# 1번 파일의 함수 호출 (검색 결과 얻기)
search_results = data_fetcher.search_and_fetch()  # (data_fetcher.py에 있는 검색 관련 함수 호출)

# 2번 파일의 함수 호출 (큐레이션 생성)
curated_results = curation_generator.curate_data(search_results)  # (curation_generator.py에서 큐레이션 생성 함수 호출)

# 최종 출력
print(curated_results)  # 큐레이션된 결과를 출력
