package middleware

import (
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// Auth middleware validates API keys
func Auth(validKeys []string) gin.HandlerFunc {
	keySet := make(map[string]struct{})
	for _, key := range validKeys {
		keySet[key] = struct{}{}
	}

	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")

		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": gin.H{
					"message": "Missing Authorization header",
					"type":    "invalid_request_error",
				},
			})
			return
		}

		// Extract bearer token
		parts := strings.SplitN(authHeader, " ", 2)
		if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": gin.H{
					"message": "Invalid Authorization header format",
					"type":    "invalid_request_error",
				},
			})
			return
		}

		token := parts[1]

		// Validate token
		if _, valid := keySet[token]; !valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": gin.H{
					"message": "Invalid API key",
					"type":    "authentication_error",
				},
			})
			return
		}

		// Store API key for rate limiting
		c.Set("api_key", token)
		c.Next()
	}
}

// RateLimit middleware implements token bucket rate limiting
func RateLimit(requestsPerWindow int, window time.Duration) gin.HandlerFunc {
	limiters := &sync.Map{}
	ratePerSecond := float64(requestsPerWindow) / window.Seconds()

	return func(c *gin.Context) {
		// Use API key or IP as rate limit key
		key := c.GetString("api_key")
		if key == "" {
			key = c.ClientIP()
		}

		// Get or create limiter
		limiterI, _ := limiters.LoadOrStore(key, rate.NewLimiter(rate.Limit(ratePerSecond), requestsPerWindow))
		limiter := limiterI.(*rate.Limiter)

		if !limiter.Allow() {
			c.Header("X-RateLimit-Remaining", "0")
			c.Header("X-RateLimit-Reset", time.Now().Add(window).Format(time.RFC3339))
			c.Header("Retry-After", "60")

			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": gin.H{
					"message": "Rate limit exceeded",
					"type":    "rate_limit_error",
				},
			})
			return
		}

		// Set rate limit headers
		remaining := int(limiter.Tokens())
		if remaining < 0 {
			remaining = 0
		}
		c.Header("X-RateLimit-Remaining", string(rune(remaining)))

		c.Next()
	}
}
