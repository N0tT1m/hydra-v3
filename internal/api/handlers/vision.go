package handlers

import (
	"net/http"

	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/gin-gonic/gin"
)

// Vision and image-generation surfaces are routed but not implemented.
//
// They answer 501 rather than a plausible-looking body so a client fails fast
// instead of building on a fabricated response. Implementing them is not a
// matter of wiring: captioning needs a vision-language model (the partial
// loader handles text decoder architectures only), and image generation needs
// a diffusion stack that does not exist in the worker yet.

func notImplemented(message string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusNotImplemented, gin.H{
			"error": gin.H{
				"message": message,
				"type":    "not_implemented",
			},
		})
	}
}

// VisionCaption handles POST /v1/vision/caption.
func VisionCaption(coord *coordinator.Coordinator) gin.HandlerFunc {
	return notImplemented("vision captioning not implemented: no vision-language model support in the worker")
}

// VisionValidate handles POST /v1/vision/validate.
func VisionValidate(coord *coordinator.Coordinator) gin.HandlerFunc {
	return notImplemented("vision validation not implemented: no vision-language model support in the worker")
}

// VisionVerify handles POST /v1/vision/verify.
func VisionVerify(coord *coordinator.Coordinator) gin.HandlerFunc {
	return notImplemented("vision verification not implemented: no vision-language model support in the worker")
}

// ImageGenerate handles POST /v1/images/generate.
func ImageGenerate(coord *coordinator.Coordinator) gin.HandlerFunc {
	return notImplemented("image generation not implemented: no diffusion pipeline in the worker")
}
