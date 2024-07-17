FROM frolvlad/alpine-gxx
COPY . /app
WORKDIR /app
RUN mkdir -p dist && \
    g++ -std=c++17 -Isrc src/*.cpp -o dist/basic_example examples/basic_example/basic_example.cpp
CMD ["./dist/basic_example"]

