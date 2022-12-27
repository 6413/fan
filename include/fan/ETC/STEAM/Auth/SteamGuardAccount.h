#pragma once

#include _WITCH_PATH(WITCH.h)

#include _WITCH_PATH(ETC/STEAM/Auth/SessionData.h)

typedef struct{
	/* json name: shared_secret */
	uint8_t SharedSecret[64];
	/* json name: serial_number */
	uint8_t SerialNumber[64];
	/* json name: revocation_code */
	uint8_t RevocationCode[64];
	/* json name: uri */
	uint8_t URI[64];
	/* json name: server_time */
	uint32_t ServerTime;
	/* json name: account_name */
	uint8_t AccountName[64];
	/* json name: token_gid */
	uint8_t TokenGID[64];
	/* json name: identity_secret */
	uint8_t IdentitySecret[64];
	/* json name: secret_1 */
	uint8_t Secret1[64];
	/* json name: status */
	sint32_t Status;
	/* json name: device_id */
	uint8_t DeviceID[64];

	/* Set to true if the authenticator has actually been applied to the account. */
	/* json name: fully_enrolled */
	bool FullyEnrolled;

	STEAM_Auth_SessionData_t Session;

	uint8_t steamGuardCodeTranslations[] = {50, 51, 52, 53, 54, 55, 56, 57, 66, 67, 68, 70, 71, 72, 74, 75, 77, 78, 80, 81, 82, 84, 86, 87, 88, 89};
}STEAM_Auth_GuardAccount_t;

/* put 2 as scheme parameter */
bool STEAM_Auth_GuardAccount_DeactivateAuthenticator(sint32_t scheme){
	var postData = new NameValueCollection();
	postData.Add("steamid", this.Session.SteamID.ToString());
	postData.Add("steamguard_scheme", scheme.ToString());
	postData.Add("revocation_code", this.RevocationCode);
	postData.Add("access_token", this.Session.OAuthToken);

	try
	{
		string response = SteamWeb.MobileLoginRequest(APIEndpoints.STEAMAPI_BASE + "/ITwoFactorService/RemoveAuthenticator/v0001", "POST", postData);
		var removeResponse = JsonConvert.DeserializeObject<RemoveAuthenticatorResponse>(response);

		if (removeResponse == null || removeResponse.Response == null || !removeResponse.Response.Success) return false;
		return true;
	}
	catch (Exception)
	{
		return false;
	}
}
